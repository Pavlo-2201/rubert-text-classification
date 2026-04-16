"""Предобученная модель: опубликованный русскоязычный BERT с зафиксированной ревизией.

cointegrated/rubert-tiny2 — BERT-энкодер (3 слоя, hidden 312, 12 голов, словарь 83 828,
29,2 млн параметров), дистиллированный для русского языка. Model card:
https://huggingface.co/cointegrated/rubert-tiny2 , лицензия MIT.

Почему не DeepPavlov/rubert-base-cased (180 млн параметров): в доступной среде нет GPU,
а дообучение base-модели на CPU заняло бы часы на эпоху. Модель задаётся конфигурацией,
и base-вариант запускается той же командой с другим --model-dir на GPU.

Классификационная голова (BertForSequenceClassification) в этом чекпойнте отсутствует и
инициализируется случайно: до дообучения это не классификатор, а только энкодер.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rubert_clf import data

MODEL_ID = "cointegrated/rubert-tiny2"
REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
LICENSE = "MIT"
MODEL_CARD_URL = f"https://huggingface.co/{MODEL_ID}"
MODEL_DIR = data.PROJECT_ROOT / "models" / "rubert-tiny2"

# Файлы ревизии и их идентификаторы на Hugging Face Hub:
# для LFS-файлов — SHA-256 содержимого, для обычных — git blob SHA-1.
FILES = {
    "config.json": ("git-sha1", "26aa82492df46244bd53d5a520aa1d364d8d7966"),
    "special_tokens_map.json": ("git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "tokenizer.json": ("git-sha1", "9bb4171eb4d2379a4a3ec41c21841823ee229ecc"),
    "tokenizer_config.json": ("git-sha1", "0ca66644d7807a33e4f9ac11d47b7c39e6ce15e0"),
    "vocab.txt": ("git-sha1", "211012a925fc51581997b6d3ef047bcda9a842f4"),
    "model.safetensors": ("sha256", "26ebb6db2a68593c54c74902d7a74f332da66297693f965cc9f1b0af4abf3894"),
}


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def verify_model_dir(model_dir: Path = MODEL_DIR) -> dict[str, bool]:
    """Сверяет файлы модели с ревизией REVISION. Возвращает {файл: совпал}."""
    result = {}
    for name, (kind, expected) in FILES.items():
        path = model_dir / name
        if not path.exists():
            result[name] = False
            continue
        actual = data.sha256_file(path) if kind == "sha256" else git_blob_sha1(path)
        result[name] = actual == expected
    return result


def require_model_dir(model_dir: Path = MODEL_DIR) -> Path:
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir} не найден. Запустите: python scripts/download_model.py")
    return model_dir
