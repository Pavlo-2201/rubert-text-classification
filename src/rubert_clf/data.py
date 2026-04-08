"""Источник данных и загрузка датасета.

Те же данные, что в проекте russian-text-classification: «Russian Language Toxic Comments»
(2ch.hk + pikabu.ru, 14 412 комментариев; Kaggle blackmoon/russian-language-toxic-comments,
версия 1, лицензия GPL-2.0). Kaggle требует авторизации, поэтому файл берётся из публичного
репозитория s-nlp/rudetoxifier по зафиксированному коммиту, где датасет входит последним
блоком в объединённый ru_toxic_dataset.csv. Константы совпадают с классическим проектом
и проверяются по SHA-256 файла и отпечатку содержимого.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SOURCE_REPO = "https://github.com/s-nlp/rudetoxifier"
SOURCE_COMMIT = "aef59899fb99c8da77d4bd6dc69c9c88607e7b6e"
SOURCE_PATH = "data/train/ru_toxic_dataset.csv"
SOURCE_URL = f"https://raw.githubusercontent.com/s-nlp/rudetoxifier/{SOURCE_COMMIT}/{SOURCE_PATH}"
SOURCE_SHA256 = "c2c929314de21b69cfa1b809560fa160361c3c202618158134bbcb79b52de0de"
SOURCE_FILE = RAW_DIR / "ru_toxic_dataset.csv"

# Блок Kaggle-датасета внутри объединённого файла: строки [SUBSET_START_ROW, конец).
SUBSET_START_ROW = 148_775
EXPECTED_ROWS = 14_412
EXPECTED_LABEL_COUNTS = {0: 9_586, 1: 4_826}
EXPECTED_FINGERPRINT = "5cce7860d064da67a4cf680b5b33fb0e7b2b92aed04029dda99f56fa68289b0f"

DATASET_FILE = RAW_DIR / "russian_language_toxic_comments.csv"

TEXT_COL = "comment"
LABEL_COL = "toxic"
ID_COL = "source_row"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Идентификатор текста: SHA-256 исходной строки в UTF-8 (без нормализации)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 от строк «source_row<TAB>sha256(text)<TAB>label\\n» в порядке source_row.

    Определение совпадает с классическим проектом и не зависит от формата CSV.
    """
    digest = hashlib.sha256()
    for row, text, label in df.sort_values(ID_COL)[[ID_COL, TEXT_COL, LABEL_COL]].itertuples(index=False):
        digest.update(f"{row}\t{text_sha256(text)}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def read_source_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: тексты вроде «NA» не должны превращаться в пропуски.
    return pd.read_csv(path, dtype={TEXT_COL: str}, keep_default_na=False)


def extract_subset(source: pd.DataFrame) -> pd.DataFrame:
    subset = source.iloc[SUBSET_START_ROW:].copy()
    subset.insert(0, ID_COL, range(SUBSET_START_ROW, SUBSET_START_ROW + len(subset)))
    labels = pd.to_numeric(subset[LABEL_COL], errors="raise")
    if not labels.isin([0.0, 1.0]).all():
        raise ValueError(f"Неожиданные значения меток: {sorted(labels.unique())}")
    subset[LABEL_COL] = labels.astype(int)
    return subset[[ID_COL, TEXT_COL, LABEL_COL]].reset_index(drop=True)


def load_dataset(path: Path = DATASET_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден. Запустите: python scripts/download_data.py")
    df = pd.read_csv(path, dtype={TEXT_COL: str}, keep_default_na=False)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df
