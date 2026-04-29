"""Этап 4: простой цикл дообучения rubert-tiny2.

    python scripts/finetune.py --smoke          # быстрый прогон на малой выборке (CPU, ~1 мин)
    python scripts/finetune.py                  # полный train, оценка на validation по эпохам

Test не используется.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import ToxicCommentsDataset, make_dataloader, make_length_sorted_dataloader
from rubert_clf.finetune import FinetuneConfig, build_model, make_optimizer, predict_proba, train_epoch
from rubert_clf.metrics import classification_metrics
from rubert_clf.runtime import get_device, set_seed

REPORT_DIR = data.PROJECT_ROOT / "reports" / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="256 текстов train, 256 validation, 1 эпоха")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = FinetuneConfig()
    if args.smoke:
        config.epochs = 1
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.lr = args.lr

    set_seed(config.seed)
    device = get_device(args.device)
    model_dir = model_source.require_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    parts = splits.load_splits()
    train_df, val_df = parts["train"], parts["validation"]
    if args.smoke:
        train_df = train_df.sample(256, random_state=config.seed)
        val_df = val_df.sample(256, random_state=config.seed)

    train_ds = ToxicCommentsDataset(train_df[data.TEXT_COL], train_df[data.LABEL_COL], tokenizer, config.max_length)
    val_ds = ToxicCommentsDataset(val_df[data.TEXT_COL], val_df[data.LABEL_COL], tokenizer, config.max_length)
    train_loader = make_dataloader(train_ds, tokenizer, config.batch_size, shuffle=True, seed=config.seed)
    val_loader, val_order = make_length_sorted_dataloader(val_ds, tokenizer, config.eval_batch_size)

    model = build_model(model_dir).to(device)
    optimizer = make_optimizer(model, config)
    y_val = val_df[data.LABEL_COL].values

    history = []
    for epoch in range(1, config.epochs + 1):
        stats = train_epoch(model, train_loader, optimizer, device)
        start = time.perf_counter()
        probs, val_loss = predict_proba(model, val_loader, device, val_order)
        eval_seconds = time.perf_counter() - start
        metrics = classification_metrics(y_val, (probs >= 0.5).astype(int), probs)
        row = {"epoch": epoch, "train_loss": round(stats["train_loss"], 4), "val_loss": round(val_loss, 4),
               "train_seconds": round(stats["seconds"], 1), "eval_seconds": round(eval_seconds, 1), "validation": metrics}
        history.append(row)
        print(f"epoch {epoch}: train_loss {row['train_loss']} val_loss {row['val_loss']} "
              f"val F1 {metrics['f1_toxic']} ({row['train_seconds']} с)")

    report = {
        "mode": "smoke" if args.smoke else "full",
        "model": f"{model_source.MODEL_ID}@{model_source.REVISION}",
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "config": config.to_dict(),
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
        "threshold": 0.5,
        "history": history,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"stage4_finetune_{report['mode']}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
