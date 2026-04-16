"""Загрузка cointegrated/rubert-tiny2 по зафиксированной ревизии и проверка хешей файлов.

    python scripts/download_model.py                      # с Hugging Face Hub
    python scripts/download_model.py --from-dir PATH      # из уже скачанной копии (без сети)

Второй вариант нужен, если huggingface.co недоступен: скачайте ревизию на другой машине
(huggingface_hub.snapshot_download(..., revision=REVISION)) и укажите папку. В обоих случаях
файлы сверяются с хешами ревизии, поэтому подменённая или другая версия не пройдёт.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rubert_clf import model_source as ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", type=Path, help="папка с уже скачанными файлами ревизии")
    parser.add_argument("--model-dir", type=Path, default=ms.MODEL_DIR)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.from_dir:
        for name in ms.FILES:
            shutil.copyfile(args.from_dir / name, args.model_dir / name)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=ms.MODEL_ID, revision=ms.REVISION, allow_patterns=list(ms.FILES), local_dir=args.model_dir
        )

    checks = ms.verify_model_dir(args.model_dir)
    for name, ok in checks.items():
        print(f"{'OK ' if ok else 'BAD'} {name}")
    if not all(checks.values()):
        raise SystemExit(f"Файлы не совпадают с ревизией {ms.REVISION}")
    print(f"{ms.MODEL_ID}@{ms.REVISION} ({ms.LICENSE}) -> {args.model_dir}")


if __name__ == "__main__":
    main()
