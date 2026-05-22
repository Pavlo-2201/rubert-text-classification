"""Обучение итоговой модели: дообучение rubert-tiny2 на train с выбором эпохи и порога по validation.

    python train.py                       # полный режим: configs/finetune_best.json, ~4 мин на 4 CPU
    python train.py --smoke               # быстрый прогон: 256 текстов, 1 эпоха, models/smoke
    python train.py --device cuda         # на GPU (amp=auto включит float16 + GradScaler)
    python train.py --resume              # продолжить прерванное обучение с последнего чекпойнта

Результат — папка модели (по умолчанию models/rubert-toxic): веса и токенизатор в формате
Hugging Face, selection.json с эпохой, порогом и метриками validation, train_metadata.json
с версиями модели, данных и окружения. Test не используется.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from rubert_clf import data, model_source, splits
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.finetune import FinetuneConfig, build_model, fit, write_json
from rubert_clf.runtime import get_device, set_seed

DEFAULT_CONFIG = data.PROJECT_ROOT / "configs" / "finetune_best.json"
SMOKE_CONFIG = data.PROJECT_ROOT / "configs" / "smoke.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help=f"JSON с FinetuneConfig (по умолчанию {DEFAULT_CONFIG.name})")
    parser.add_argument("--smoke", action="store_true", help=f"малая выборка и 1 эпоха ({SMOKE_CONFIG.name})")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-model-dir", type=Path, default=model_source.MODEL_DIR,
                        help="предобученный энкодер; хеши проверяются только для rubert-tiny2")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="по умолчанию checkpoints/<имя папки модели>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config or (SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = FinetuneConfig.from_dict(raw)
    output_dir = args.output_dir or data.PROJECT_ROOT / "models" / ("smoke" if args.smoke else "rubert-toxic")
    checkpoint_dir = args.checkpoint_dir or data.PROJECT_ROOT / "checkpoints" / output_dir.name

    base_dir = args.base_model_dir
    if base_dir == model_source.MODEL_DIR:
        checks = model_source.verify_model_dir(model_source.require_model_dir())
        if not all(checks.values()):
            raise SystemExit(f"Файлы модели не совпадают с ревизией {model_source.REVISION}: {checks}")

    set_seed(config.seed)
    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    parts = splits.load_splits()
    train_df, val_df = parts["train"], parts["validation"]
    if "sample_train" in raw:
        train_df = train_df.sample(raw["sample_train"], random_state=config.seed)
        val_df = val_df.sample(raw["sample_validation"], random_state=config.seed)

    train_ds = ToxicCommentsDataset(train_df[data.TEXT_COL], train_df[data.LABEL_COL], tokenizer, config.max_length)
    val_ds = ToxicCommentsDataset(val_df[data.TEXT_COL], val_df[data.LABEL_COL], tokenizer, config.max_length)
    print(f"train {len(train_ds)}, validation {len(val_ds)}, устройство {device}, конфигурация {config_path.name}")
    model = build_model(base_dir)
    start = time.perf_counter()
    history = fit(model, train_ds, val_ds, tokenizer, config, device, checkpoint_dir=checkpoint_dir, resume=args.resume)
    total = time.perf_counter() - start

    best_dir = checkpoint_dir / "best"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(best_dir, output_dir)
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    write_json(output_dir / "train_metadata.json", {
        "mode": "smoke" if args.smoke else "full",
        "config_file": config_path.name,
        "base_model": f"{model_source.MODEL_ID}@{model_source.REVISION}" if base_dir == model_source.MODEL_DIR else str(base_dir),
        "data_manifest_sha256": splits.MANIFEST_SHA256,
        "data_fingerprint": data.EXPECTED_FINGERPRINT,
        "n_train": len(train_ds), "n_validation": len(val_ds),
        "device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "total_seconds": round(total, 1),
        "history": [{k: v for k, v in h.items() if k != "val_probs"} for h in history],
    })
    print(f"Выбрана эпоха {selection['epoch']}, порог {selection['threshold']:.4f}, "
          f"validation F1 {selection['validation_tuned']['f1_toxic']}")
    print(f"Модель сохранена в {output_dir}")


if __name__ == "__main__":
    main()
