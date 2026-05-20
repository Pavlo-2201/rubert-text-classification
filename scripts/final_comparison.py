"""Этап 7: финальное сравнение трёх подходов на test и анализ ошибок на тех же текстах, что у TF-IDF.

Подходы (все обучены только на train, все параметры и пороги выбраны без test):
1. TF-IDF char_wb 2–5 + LinearSVC — конфигурация и порог из классического проекта;
2. замороженный rubert-tiny2 + LogReg — конфигурация и порог из этапа 3;
3. дообученный rubert-tiny2 — лучший прогон и эпоха этапа 6 (checkpoints/stage6/<run>/best).

Test используется один раз: модели и пороги уже зафиксированы.

Сохраняет:
- reports/final_comparison.json — метрики test, бутстреп-интервалы, парные тесты, время;
- reports/test_predictions_all.csv — попримерные оценки трёх моделей на test;
- reports/error_analysis_rubert.json — доли ошибок по группам текстов на validation (как в классике)
  и пересечение ошибок с TF-IDF;
- reports/error_examples_rubert.csv — реальные примеры, где модели расходятся, и 22 размеченных
  примера классического проекта с оценками всех трёх моделей.

    python scripts/final_comparison.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader
from rubert_clf.embeddings import extract_embeddings
from rubert_clf.errors import categorize, oov_share, train_vocabulary
from rubert_clf.finetune import predict_proba, write_json
from rubert_clf.metrics import bootstrap_f1_ci, classification_metrics, mcnemar, paired_bootstrap_f1_diff
from rubert_clf.runtime import get_device, set_seed
from rubert_clf.tfidf_baseline import CLASSICAL_DIR, build_classical_baseline

REPORTS = data.PROJECT_ROOT / "reports"
STAGE3 = REPORTS / "experiments" / "stage3_frozen_encoder.json"
STAGE6 = REPORTS / "experiments" / "stage6"
SEED = 42
LATENCY_TEXTS = 200


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def latency_ms(predict_one, texts) -> float:
    predict_one(texts[0])  # прогрев
    start = time.perf_counter()
    for t in texts:
        predict_one(t)
    return 1000 * (time.perf_counter() - start) / len(texts)


def class_separation(cls_emb: np.ndarray, y: np.ndarray) -> dict:
    """Косинус между центроидами классов и средний косинус текста к центроиду своего класса."""
    unit = cls_emb / np.linalg.norm(cls_emb, axis=1, keepdims=True)
    centroids = [unit[y == c].mean(0) for c in (0, 1)]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    within = np.mean([cos(u, centroids[c]) for u, c in zip(unit, y)])
    return {"centroid_cosine": round(cos(*centroids), 4), "mean_cosine_to_own_centroid": round(float(within), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device(args.device)
    parts = splits.load_splits()
    train, val, test = parts["train"], parts["validation"], parts["test"]
    y_train, y_val, y_test = (df[data.LABEL_COL].values for df in (train, val, test))
    results: dict = {"device": str(device), "torch_threads": torch.get_num_threads(), "n_test": len(test), "models": {}}
    latency_sample = test[data.TEXT_COL].iloc[:LATENCY_TEXTS].tolist()

    # 1. TF-IDF + LinearSVC
    tfidf = build_classical_baseline()
    _, fit_s = timed(lambda: tfidf.fit(train[data.TEXT_COL], y_train))
    tfidf_test_scores, infer_s = timed(lambda: tfidf.decision_function(test[data.TEXT_COL]))
    tfidf_test_pred = (tfidf_test_scores >= tfidf.threshold).astype(int)
    assert (tfidf_test_pred == tfidf.predict(test[data.TEXT_COL])).all()
    tfidf_val_scores, tfidf_val_pred = tfidf.decision_function(val[data.TEXT_COL]), tfidf.predict(val[data.TEXT_COL])
    classical_test = pd.read_csv(CLASSICAL_DIR / "test_predictions.csv")
    results["models"]["tfidf_linearsvc"] = {
        "threshold": float(tfidf.threshold), "threshold_on": "decision_function",
        "train_seconds": round(fit_s, 2), "test_inference_seconds": round(infer_s, 2),
        "latency_ms_per_text": round(latency_ms(lambda t: tfidf.predict([t]), latency_sample), 2),
        "matches_classical_test_predictions": bool(
            (classical_test.set_index(data.ID_COL).loc[test[data.ID_COL], "pred"].values == tfidf_test_pred).all()),
    }

    # 2. Замороженный encoder + LogReg (конфигурация этапа 3)
    stage3 = json.loads(STAGE3.read_text(encoding="utf-8"))
    best3, frozen_threshold = stage3["best"], stage3["threshold"]["used"]
    tokenizer = AutoTokenizer.from_pretrained(model_source.require_model_dir())
    encoder = AutoModel.from_pretrained(model_source.MODEL_DIR)
    for p in encoder.parameters():
        p.requires_grad_(False)
    L3, pooling = best3["max_length"], best3["pooling"]

    def embed(df):
        ds = ToxicCommentsDataset(df[data.TEXT_COL], None, tokenizer, L3)
        return extract_embeddings(encoder, ds, tokenizer, device)[pooling]

    probe = make_pipeline(StandardScaler(), LogisticRegression(C=best3["C"], class_weight=best3["class_weight"],
                                                               max_iter=5000, random_state=SEED))
    X_train, extract_s = timed(lambda: embed(train))
    _, probe_s = timed(lambda: probe.fit(X_train, y_train))
    frozen_test_probs, frozen_infer_s = timed(lambda: probe.predict_proba(embed(test))[:, 1])
    frozen_val_probs = probe.predict_proba(embed(val))[:, 1]
    frozen_one = lambda t: probe.predict_proba(
        extract_embeddings(encoder, ToxicCommentsDataset([t], None, tokenizer, L3), tokenizer, device, 1)[pooling])
    results["models"]["frozen_rubert_logreg"] = {
        "config": best3, "threshold": frozen_threshold, "threshold_on": "probability",
        "train_seconds": round(extract_s + probe_s, 2),
        "train_seconds_detail": {"embedding_extraction": round(extract_s, 2), "logreg_fit": round(probe_s, 2)},
        "test_inference_seconds": round(frozen_infer_s, 2),
        "latency_ms_per_text": round(latency_ms(frozen_one, latency_sample), 2),
    }

    # 3. Дообученный rubert-tiny2 (лучший прогон этапа 6)
    summary6 = json.loads((STAGE6 / "summary.json").read_text(encoding="utf-8"))
    run = summary6["chosen_run"]
    best_dir = data.PROJECT_ROOT / "checkpoints" / "stage6" / run / "best"
    selection = json.loads((best_dir / "selection.json").read_text(encoding="utf-8"))
    run_report = json.loads((STAGE6 / f"{run}.json").read_text(encoding="utf-8"))
    ft_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    ft_threshold, L6 = selection["threshold"], selection["config"]["max_length"]

    def ft_probs(df, batch_size=128):
        ds = ToxicCommentsDataset(df[data.TEXT_COL], None, tokenizer, L6)
        loader, order = make_length_sorted_dataloader(ds, tokenizer, batch_size)
        return predict_proba(ft_model, loader, device, order)[0]

    ft_test_probs, ft_infer_s = timed(lambda: ft_probs(test))
    ft_val_probs = ft_probs(val)
    stored_val = np.asarray(run_report["best_val_probs"])
    epochs_to_best = [h for h in run_report["history"] if h["epoch"] <= selection["epoch"]]
    results["models"]["finetuned_rubert"] = {
        "run": run, "epoch": selection["epoch"], "config": selection["config"],
        "threshold": ft_threshold, "threshold_on": "probability",
        "train_seconds": round(sum(h["train_seconds"] for h in epochs_to_best), 1),
        "train_seconds_note": "сумма эпох до выбранной; полный прогон с оценкой validation — total_seconds",
        "total_run_seconds": run_report["total_seconds"],
        "test_inference_seconds": round(ft_infer_s, 2),
        "latency_ms_per_text": round(latency_ms(lambda t: ft_probs(pd.DataFrame({data.TEXT_COL: [t]}), 1), latency_sample), 2),
        "validation_probs_match_stage6": bool(np.allclose(ft_val_probs, stored_val, atol=1e-4)),
    }

    # Метрики test
    preds = {
        "tfidf_linearsvc": (tfidf_test_pred, tfidf_test_scores),
        "frozen_rubert_logreg": ((frozen_test_probs >= frozen_threshold).astype(int), frozen_test_probs),
        "finetuned_rubert": ((ft_test_probs >= ft_threshold).astype(int), ft_test_probs),
    }
    for name, (pred, scores) in preds.items():
        results["models"][name]["test"] = classification_metrics(y_test, pred, scores)
        results["models"][name]["test_f1_bootstrap_ci"] = bootstrap_f1_ci(y_test, pred)
        results["models"][name]["test_inference_ms_per_1000"] = round(
            1000 * results["models"][name]["test_inference_seconds"] / len(test) * 1000, 1)
    results["paired_tests"] = {
        f"{a}_vs_{b}": {"mcnemar": mcnemar(y_test, preds[a][0], preds[b][0]),
                        "f1_diff_bootstrap": paired_bootstrap_f1_diff(y_test, preds[a][0], preds[b][0])}
        for a, b in [("finetuned_rubert", "tfidf_linearsvc"), ("frozen_rubert_logreg", "tfidf_linearsvc"),
                     ("finetuned_rubert", "frozen_rubert_logreg")]
    }
    pd.DataFrame({
        data.ID_COL: test[data.ID_COL], data.LABEL_COL: y_test,
        "tfidf_score": tfidf_test_scores.round(6), "tfidf_pred": preds["tfidf_linearsvc"][0],
        "frozen_prob": frozen_test_probs.round(6), "frozen_pred": preds["frozen_rubert_logreg"][0],
        "finetuned_prob": ft_test_probs.round(6), "finetuned_pred": preds["finetuned_rubert"][0],
    }).to_csv(REPORTS / "test_predictions_all.csv", index=False)

    # Анализ ошибок на validation — тех же текстах, что в классическом проекте
    vocab = train_vocabulary(train[data.TEXT_COL])
    cats = pd.DataFrame([categorize(t, oov_share(t, vocab)) for t in val[data.TEXT_COL]])
    val_preds = {
        "tfidf_linearsvc": tfidf_val_pred,
        "frozen_rubert_logreg": (frozen_val_probs >= frozen_threshold).astype(int),
        "finetuned_rubert": (ft_val_probs >= ft_threshold).astype(int),
    }
    groups = {"all": np.ones(len(val), bool), **{c: cats[c].values for c in cats.columns}}
    by_group = {}
    for g, mask in groups.items():
        neg, pos = mask & (y_val == 0), mask & (y_val == 1)
        by_group[g] = {"n": int(mask.sum()), "n_toxic": int(pos.sum())}
        for name, pred in val_preds.items():
            by_group[g][name] = {"fp_rate": round(float((pred[neg] == 1).mean()), 3) if neg.any() else None,
                                 "fn_rate": round(float((pred[pos] == 0).mean()), 3) if pos.any() else None}
    classical_errors = pd.read_csv(CLASSICAL_DIR / "validation_errors.csv")
    tfidf_wrong = val_preds["tfidf_linearsvc"] != y_val
    ft_wrong = val_preds["finetuned_rubert"] != y_val
    frozen_wrong = val_preds["frozen_rubert_logreg"] != y_val
    overlap = {
        "tfidf_errors": int(tfidf_wrong.sum()),
        "tfidf_errors_match_classical_file": set(val[data.ID_COL][tfidf_wrong]) == set(classical_errors[data.ID_COL]),
        "finetuned_errors": int(ft_wrong.sum()),
        "frozen_errors": int(frozen_wrong.sum()),
        "both_tfidf_and_finetuned_wrong": int((tfidf_wrong & ft_wrong).sum()),
        "fixed_by_finetuned": int((tfidf_wrong & ~ft_wrong).sum()),
        "new_errors_of_finetuned": int((~tfidf_wrong & ft_wrong).sum()),
        "all_three_wrong": int((tfidf_wrong & ft_wrong & frozen_wrong).sum()),
    }
    test_overlap = {
        "tfidf_errors": int((preds["tfidf_linearsvc"][0] != y_test).sum()),
        "finetuned_errors": int((preds["finetuned_rubert"][0] != y_test).sum()),
        "both_wrong": int(((preds["tfidf_linearsvc"][0] != y_test) & (preds["finetuned_rubert"][0] != y_test)).sum()),
    }

    # Как дообучение меняет представление [CLS] (validation)
    with torch.no_grad():
        ds = ToxicCommentsDataset(val[data.TEXT_COL], None, tokenizer, L6)
        loader, order = make_length_sorted_dataloader(ds, tokenizer, 128)
        cls_ft = []
        ft_model.eval()
        for batch in loader:
            hidden = ft_model.bert(input_ids=batch["input_ids"].to(device),
                                   attention_mask=batch["attention_mask"].to(device)).last_hidden_state
            cls_ft.append(hidden[:, 0].cpu().numpy())
        cls_ft_sorted = np.concatenate(cls_ft)
        cls_ft = np.empty_like(cls_ft_sorted)
        cls_ft[order] = cls_ft_sorted
    cls_frozen = extract_embeddings(encoder, ToxicCommentsDataset(val[data.TEXT_COL], None, tokenizer, L6), tokenizer, device)["cls"]
    separation = {"frozen_encoder": class_separation(cls_frozen, y_val), "finetuned_encoder": class_separation(cls_ft, y_val)}

    val_table = pd.DataFrame({
        data.ID_COL: val[data.ID_COL], data.LABEL_COL: y_val,
        "tfidf_score": tfidf_val_scores.round(3), "tfidf_pred": val_preds["tfidf_linearsvc"],
        "frozen_prob": frozen_val_probs.round(3), "frozen_pred": val_preds["frozen_rubert_logreg"],
        "finetuned_prob": ft_val_probs.round(3), "finetuned_pred": val_preds["finetuned_rubert"],
        **{c: cats[c].values for c in cats.columns}, data.TEXT_COL: val[data.TEXT_COL].str.strip(),
    })
    annotated = pd.read_csv(CLASSICAL_DIR / "error_examples.csv", keep_default_na=False)
    examples = annotated[[data.ID_COL, "error", "category", "explanation"]].merge(
        val_table.drop(columns=list(cats.columns)), on=data.ID_COL, how="left", validate="one_to_one")
    examples.insert(1, "source", "classical_annotated")
    fixed = val_table[tfidf_wrong & ~ft_wrong].assign(margin=lambda d: (d.finetuned_prob - ft_threshold).abs())
    new = val_table[~tfidf_wrong & ft_wrong].assign(margin=lambda d: (d.finetuned_prob - ft_threshold).abs())
    extra = pd.concat([
        fixed.nlargest(10, "margin").assign(source="fixed_by_finetuned"),
        new.nlargest(10, "margin").assign(source="new_error_of_finetuned"),
    ]).drop(columns=["margin", *cats.columns])
    pd.concat([examples, extra], ignore_index=True).to_csv(REPORTS / "error_examples_rubert.csv", index=False)
    val_table.drop(columns=[data.TEXT_COL]).to_csv(REPORTS / "validation_predictions_all.csv", index=False)

    write_json(REPORTS / "error_analysis_rubert.json", {
        "split": "validation",
        "groups_note": "эвристические группы как в russian-text-classification (src/rubert_clf/errors.py)",
        "by_group": by_group, "error_overlap_validation": overlap, "error_overlap_test": test_overlap,
        "cls_separation": separation,
    })
    write_json(REPORTS / "final_comparison.json", results)
    print(json.dumps({k: {m: v.get("test", {}).get("f1_toxic") for m, v in results["models"].items()} for k in ["f1"]},
                     ensure_ascii=False))
    print(json.dumps(results["paired_tests"], ensure_ascii=False, indent=1))
    print(json.dumps(overlap, ensure_ascii=False), json.dumps(separation, ensure_ascii=False))


if __name__ == "__main__":
    main()
