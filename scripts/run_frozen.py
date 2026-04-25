"""Этап 3: замороженный rubert-tiny2 как извлекатель эмбеддингов + логистическая регрессия.

Протокол как в классическом проекте: конфигурация (пулинг, max_length, C, class_weight)
выбирается 5-fold CV на train, порог — по out-of-fold вероятностям на train, validation
служит для сравнения с TF-IDF baseline. Test здесь не используется.

Эмбеддинги кешируются в artifacts/embeddings/ (не в Git).

    python scripts/run_frozen.py [--max-lengths 128 512] [--limit-train N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.embeddings import POOLINGS, extract_embeddings
from rubert_clf.metrics import best_f1_threshold, classification_metrics
from rubert_clf.runtime import get_device, set_seed
from rubert_clf.tfidf_baseline import build_classical_baseline

SEED = 42
CACHE_DIR = data.PROJECT_ROOT / "artifacts" / "embeddings"
REPORT_DIR = data.PROJECT_ROOT / "reports" / "experiments"
C_GRID = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")


def build_probe(C: float, class_weight) -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=SEED)
    )


def get_embeddings(parts, tokenizer, encoder, device, max_length: int) -> tuple[dict, dict]:
    """{split: {pooling: array}} и время извлечения; кеш проверяется по source_row."""
    embeddings, timings = {}, {}
    for name, df in parts.items():
        path = CACHE_DIR / f"{name}_L{max_length}.npz"
        if path.exists():
            cached = np.load(path)
            if np.array_equal(cached["source_row"], df[data.ID_COL].values):
                embeddings[name] = {p: cached[p] for p in POOLINGS}
                timings[name] = float(cached["seconds"])
                continue
        ds = ToxicCommentsDataset(df[data.TEXT_COL], None, tokenizer, max_length)
        start = time.perf_counter()
        embeddings[name] = extract_embeddings(encoder, ds, tokenizer, device)
        timings[name] = time.perf_counter() - start
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, source_row=df[data.ID_COL].values, seconds=timings[name], **embeddings[name])
    return embeddings, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(SEED)
    torch.set_num_threads(torch.get_num_threads())
    device = get_device(args.device)
    model_dir = model_source.require_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    for param in encoder.parameters():
        param.requires_grad_(False)  # замороженный энкодер

    parts = {k: v for k, v in splits.load_splits().items() if k in ("train", "validation")}
    y_train = parts["train"][data.LABEL_COL].values
    y_val = parts["validation"][data.LABEL_COL].values

    all_embeddings, extraction = {}, {}
    for max_length in args.max_lengths:
        all_embeddings[max_length], timings = get_embeddings(parts, tokenizer, encoder, device, max_length)
        extraction[str(max_length)] = {k: round(v, 1) for k, v in timings.items()}
        print(f"max_length={max_length}: извлечение {extraction[str(max_length)]} с")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = []
    for max_length, pooling, C, cw in itertools.product(args.max_lengths, POOLINGS, C_GRID, CLASS_WEIGHTS):
        X = all_embeddings[max_length]["train"][pooling]
        oof = cross_val_predict(build_probe(C, cw), X, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        fold_f1 = []
        for _, idx in cv.split(X, y_train):
            fold_f1.append(classification_metrics(y_train[idx], (oof[idx] >= 0.5).astype(int))["f1_toxic"])
        row = {
            "max_length": max_length, "pooling": pooling, "C": C, "class_weight": cw,
            "cv_f1_toxic_mean": round(float(np.mean(fold_f1)), 4), "cv_f1_toxic_std": round(float(np.std(fold_f1)), 4),
            "_oof": oof,
        }
        grid.append(row)
        print({k: v for k, v in row.items() if k != "_oof"})

    best = max(grid, key=lambda r: r["cv_f1_toxic_mean"])
    threshold, oof_f1 = best_f1_threshold(y_train, best["_oof"])
    probe = build_probe(best["C"], best["class_weight"])
    start = time.perf_counter()
    probe.fit(all_embeddings[best["max_length"]]["train"][best["pooling"]], y_train)
    probe_fit_seconds = time.perf_counter() - start
    val_scores = probe.predict_proba(all_embeddings[best["max_length"]]["validation"][best["pooling"]])[:, 1]
    val_default = classification_metrics(y_val, (val_scores >= 0.5).astype(int), val_scores)
    val_tuned = classification_metrics(y_val, (val_scores >= threshold).astype(int), val_scores)
    used = threshold if val_tuned["f1_toxic"] >= val_default["f1_toxic"] else 0.5

    # TF-IDF baseline тем же кодом на тех же разбиениях
    baseline = build_classical_baseline()
    start = time.perf_counter()
    baseline.fit(parts["train"][data.TEXT_COL], y_train)
    tfidf_fit_seconds = time.perf_counter() - start
    tfidf_scores = baseline.decision_function(parts["validation"][data.TEXT_COL])
    tfidf_val = classification_metrics(y_val, baseline.predict(parts["validation"][data.TEXT_COL]), tfidf_scores)
    classical_reported = json.loads((data.PROJECT_ROOT / "reference" / "classical" / "best_model.json").read_text())["validation"]["used_threshold"]

    report = {
        "model": f"{model_source.MODEL_ID}@{model_source.REVISION}",
        "device": str(device),
        "protocol": "5-fold CV на train (F1 при пороге 0.5) -> порог по OOF на train -> validation",
        "extraction_seconds": extraction,
        "grid": [{k: v for k, v in r.items() if k != "_oof"} for r in grid],
        "best": {k: v for k, v in best.items() if k != "_oof"},
        "threshold": {"oof_tuned": round(threshold, 4), "oof_f1_toxic": round(oof_f1, 4), "used": round(used, 4),
                      "rule": "подобранный порог принимается, если F1 на validation не ниже, чем при 0.5"},
        "probe_fit_seconds": round(probe_fit_seconds, 2),
        "validation": {"threshold_0.5": val_default, "used_threshold": val_tuned if used != 0.5 else val_default},
        "tfidf_baseline": {
            "validation": tfidf_val,
            "fit_seconds": round(tfidf_fit_seconds, 2),
            "matches_classical_report": tfidf_val["confusion_matrix"] == classical_reported["confusion_matrix"],
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage3_frozen_encoder.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best", "threshold", "validation", "tfidf_baseline")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
