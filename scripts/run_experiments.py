"""Этап 6: выбор checkpoint по validation, дисбаланс, learning rate и max_length.

Сетка задаётся в configs/stage6_experiments.json:
1) lr из lr_grid при базовой конфигурации;
2) при лучшем lr — варианты из at_best_lr (class_weight=balanced, max_length 64/256).
В каждом прогоне лучшая эпоха и порог выбираются по validation (fit сохраняет её в best/).
Итоговая конфигурация — лучшая по F1 на validation при подобранном пороге; она записывается
в configs/finetune_best.json. Test не используется.

Готовые прогоны (есть JSON в папке отчётов) не перезапускаются.

    python scripts/run_experiments.py            # полный режим, ~35 мин на 4 CPU
    python scripts/run_experiments.py --smoke    # 256 текстов, 1 эпоха, 2 значения lr
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.finetune import FinetuneConfig, build_model, fit, write_json
from rubert_clf.metrics import classification_metrics
from rubert_clf.runtime import get_device, set_seed

GRID_PATH = data.PROJECT_ROOT / "configs" / "stage6_experiments.json"
BEST_CONFIG_PATH = data.PROJECT_ROOT / "configs" / "finetune_best.json"
SMOKE_SIZE = 256


def run_name(config: FinetuneConfig) -> str:
    return f"lr{config.lr:g}_L{config.max_length}_cw-{config.class_weight}"


def run_one(config, parts, tokenizer, device, out_dir, ckpt_root, keep_last: bool) -> dict:
    name = run_name(config)
    report_path = out_dir / f"{name}.json"
    if report_path.exists():
        print(f"{name}: уже есть отчёт, пропускаю")
        return json.loads(report_path.read_text(encoding="utf-8"))
    print(f"=== {name}")
    set_seed(config.seed)
    train_df, val_df = parts["train"], parts["validation"]
    train_ds = ToxicCommentsDataset(train_df[data.TEXT_COL], train_df[data.LABEL_COL], tokenizer, config.max_length)
    val_ds = ToxicCommentsDataset(val_df[data.TEXT_COL], val_df[data.LABEL_COL], tokenizer, config.max_length)
    model = build_model(model_source.MODEL_DIR)
    start = time.perf_counter()
    history = fit(model, train_ds, val_ds, tokenizer, config, device, checkpoint_dir=ckpt_root / name)
    total = time.perf_counter() - start
    if not keep_last:
        (ckpt_root / name / "last.pt").unlink(missing_ok=True)
    best = max(history, key=lambda h: h["validation_tuned"]["f1_toxic"])
    report = {
        "run": name,
        "config": config.to_dict(),
        "device": str(device),
        "total_seconds": round(total, 1),
        "train_truncated_share": round(train_ds.truncated_share, 4),
        "best_epoch": best["epoch"],
        "best_threshold": best["val_threshold"],
        "best_validation_tuned": best["validation_tuned"],
        "best_validation_at_0.5": best["validation"],
        "history": [{k: v for k, v in h.items() if k != "val_probs"} for h in history],
        "best_val_probs": best["val_probs"],
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-last", action="store_true", help="не удалять last.pt после прогона")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    base = FinetuneConfig.from_dict(grid["base"])
    lr_grid = grid["lr_grid"]
    if args.smoke:
        base.epochs = 1
        lr_grid = lr_grid[1:3]
    out_dir = data.PROJECT_ROOT / "reports" / "experiments" / ("stage6_smoke" if args.smoke else "stage6")
    ckpt_root = data.PROJECT_ROOT / "checkpoints" / ("stage6_smoke" if args.smoke else "stage6")

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_source.require_model_dir())
    parts = splits.load_splits()
    if args.smoke:
        parts = {k: v.sample(SMOKE_SIZE, random_state=base.seed).reset_index(drop=True) for k, v in parts.items()}

    reports = []
    for lr in lr_grid:
        reports.append(run_one(FinetuneConfig.from_dict({**base.to_dict(), "lr": lr}), parts, tokenizer, device,
                               out_dir, ckpt_root, args.keep_last))
    best_lr = max(reports, key=lambda r: r["best_validation_tuned"]["f1_toxic"])["config"]["lr"]
    print(f"Лучший lr по validation: {best_lr:g}")
    for override in grid["at_best_lr"]:
        config = FinetuneConfig.from_dict({**base.to_dict(), "lr": best_lr, **override})
        reports.append(run_one(config, parts, tokenizer, device, out_dir, ckpt_root, args.keep_last))

    # Сводка и влияние truncation на длинных текстах validation
    val = parts["validation"]
    y_val = val[data.LABEL_COL].values
    full_lengths = np.asarray(ToxicCommentsDataset(val[data.TEXT_COL], None, tokenizer, 4096).full_lengths)
    predictions = pd.DataFrame({data.ID_COL: val[data.ID_COL], data.LABEL_COL: y_val, "n_tokens": full_lengths})
    rows = []
    for r in reports:
        probs = np.asarray(r["best_val_probs"])
        predictions[r["run"]] = probs.round(6)
        pred = (probs >= r["best_threshold"]).astype(int)
        long_metrics = {}
        for min_len in (64, 128):
            mask = full_lengths > min_len
            long_metrics[f"f1_toxic_tokens_gt_{min_len}"] = classification_metrics(y_val[mask], pred[mask])["f1_toxic"]
            long_metrics[f"n_tokens_gt_{min_len}"] = int(mask.sum())
        rows.append({
            "run": r["run"], "lr": r["config"]["lr"], "max_length": r["config"]["max_length"],
            "class_weight": r["config"]["class_weight"], "best_epoch": r["best_epoch"],
            "threshold": r["best_threshold"], **{f"val_{k}": v for k, v in r["best_validation_tuned"].items() if k != "confusion_matrix"},
            "val_f1_at_0.5": r["best_validation_at_0.5"]["f1_toxic"], **long_metrics,
            "epoch_seconds_mean": round(float(np.mean([h["train_seconds"] for h in r["history"]])), 1),
        })
    table = pd.DataFrame(rows)
    chosen = max(reports, key=lambda r: r["best_validation_tuned"]["f1_toxic"])
    summary = {
        "selection": grid["selection"],
        "best_lr_stage_a": best_lr,
        "chosen_run": chosen["run"],
        "chosen_config": chosen["config"],
        "chosen_epoch": chosen["best_epoch"],
        "chosen_threshold": chosen["best_threshold"],
        "runs": rows,
    }
    write_json(out_dir / "summary.json", summary)
    table.to_csv(out_dir / "summary.csv", index=False)
    predictions.to_csv(out_dir / "val_predictions.csv", index=False)
    print(table.to_string(index=False))
    if not args.smoke:
        write_json(BEST_CONFIG_PATH, {**chosen["config"], "epochs": chosen["config"]["epochs"],
                                      "_selected_by": "scripts/run_experiments.py", "_run": chosen["run"],
                                      "_best_epoch": chosen["best_epoch"], "_threshold": chosen["best_threshold"]})
        print(f"Выбрано: {chosen['run']}, эпоха {chosen['best_epoch']}, порог {chosen['best_threshold']:.4f}")
    if args.smoke:
        shutil.rmtree(ckpt_root, ignore_errors=True)


if __name__ == "__main__":
    main()
