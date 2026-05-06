"""Дообучение rubert-tiny2 с оценкой на validation после каждой эпохи.

    python scripts/finetune.py --smoke                 # малая выборка, 1 эпоха, CPU ~ 10 с
    python scripts/finetune.py --out reports/experiments/run.json [--lr 5e-5 --epochs 3 ...]
    python scripts/finetune.py ... --resume            # продолжить с checkpoints/<run>/last.pt

Test не используется.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.finetune import FinetuneConfig, build_model, fit, write_json
from rubert_clf.runtime import get_device, set_seed

SMOKE_SIZE = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_SIZE} текстов train и validation, 1 эпоха")
    parser.add_argument("--run-name", default=None, help="имя прогона (папка чекпойнтов)")
    parser.add_argument("--out", default=None, help="куда сохранить отчёт JSON")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    for field, default in FinetuneConfig().to_dict().items():
        if isinstance(default, bool):
            parser.add_argument(f"--{field.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=None)
    args = parser.parse_args()

    config = FinetuneConfig()
    if args.smoke:
        config.epochs = 1
    for field in config.to_dict():
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)

    mode = "smoke" if args.smoke else "full"
    run_name = args.run_name or f"{mode}_lr{config.lr:g}_L{config.max_length}"
    set_seed(config.seed)
    device = get_device(args.device)
    model_dir = model_source.require_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    parts = splits.load_splits()
    train_df, val_df = parts["train"], parts["validation"]
    if args.smoke:
        train_df = train_df.sample(SMOKE_SIZE, random_state=config.seed)
        val_df = val_df.sample(SMOKE_SIZE, random_state=config.seed)

    train_ds = ToxicCommentsDataset(train_df[data.TEXT_COL], train_df[data.LABEL_COL], tokenizer, config.max_length)
    val_ds = ToxicCommentsDataset(val_df[data.TEXT_COL], val_df[data.LABEL_COL], tokenizer, config.max_length)
    model = build_model(model_dir)
    start = time.perf_counter()
    history = fit(model, train_ds, val_ds, tokenizer, config, device,
                  checkpoint_dir=data.PROJECT_ROOT / "checkpoints" / run_name, resume=args.resume)
    report = {
        "mode": mode,
        "run_name": run_name,
        "model": f"{model_source.MODEL_ID}@{model_source.REVISION}",
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "config": config.to_dict(),
        "n_train": len(train_ds),
        "n_validation": len(val_ds),
        "total_seconds": round(time.perf_counter() - start, 1),
        "history": [{k: v for k, v in row.items() if k != "val_probs"} for row in history],
    }
    out = Path(args.out) if args.out else Path("reports") / "experiments" / f"finetune_{run_name}.json"
    write_json(out if out.is_absolute() else data.PROJECT_ROOT / out, report)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
