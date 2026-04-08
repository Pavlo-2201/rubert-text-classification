"""Разбиения train/validation/test, общие с классическим проектом.

Источник истины — манифест data/splits/split_manifest.csv, скопированный из
russian-text-classification без изменений (SHA-256 зафиксирован ниже). В нём нет текстов,
только source_row, SHA-256 текста, SHA-256 ключа дедупликации, метка и разбиение.

Для проверки манифест можно построить заново из данных (rebuild_manifest): правило
дедупликации и разбиения повторяет классический проект. Совпадение двух манифестов
построчно доказывает, что используются те же данные и те же разбиения.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from rubert_clf import data

SPLITS_DIR = data.PROJECT_ROOT / "data" / "splits"
MANIFEST_PATH = SPLITS_DIR / "split_manifest.csv"
FINGERPRINT_PATH = SPLITS_DIR / "data_fingerprint.json"
MANIFEST_SHA256 = "a5c21437095942fde727b0775624660d0d083cd411348594c449305a704b23de"

SPLITS = ("train", "validation", "test")
EXCLUDED = "excluded"
SPLIT_SEED = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

_WHITESPACE = re.compile(r"\s+")


def load_manifest(path: Path = MANIFEST_PATH, check_sha: bool = True) -> pd.DataFrame:
    if check_sha:
        actual = data.sha256_file(path)
        if actual != MANIFEST_SHA256:
            raise ValueError(f"SHA-256 манифеста {actual} не совпадает с зафиксированным {MANIFEST_SHA256}")
    return pd.read_csv(path, dtype={"exclusion_reason": str}, keep_default_na=False)


def load_fingerprint(path: Path = FINGERPRINT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_splits(dataset: pd.DataFrame | None = None, manifest: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """{'train', 'validation', 'test'} -> DataFrame[source_row, comment, toxic] с проверкой хешей."""
    dataset = data.load_dataset() if dataset is None else dataset
    manifest = load_manifest() if manifest is None else manifest
    merged = manifest.merge(dataset, on=data.ID_COL, how="left", suffixes=("", "_dataset"), validate="one_to_one")
    if merged[data.TEXT_COL].isna().any():
        raise ValueError("В датасете нет части строк манифеста")
    if not (merged[data.TEXT_COL].map(data.text_sha256) == merged["text_sha256"]).all():
        raise ValueError("Хеши текстов не совпадают с манифестом: данные отличаются")
    if not (merged[data.LABEL_COL] == merged[f"{data.LABEL_COL}_dataset"]).all():
        raise ValueError("Метки не совпадают с манифестом")
    cols = [data.ID_COL, data.TEXT_COL, data.LABEL_COL]
    return {name: merged.loc[merged["split"] == name, cols].reset_index(drop=True) for name in SPLITS}


# --- Независимое построение манифеста для проверки -------------------------------------

def duplicate_key(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def rebuild_manifest(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Повторяет правило классического проекта: дедупликация по ключу, исключение групп
    с противоречивыми метками, стратифицированное разбиение 70/15/15 по уникальным текстам."""
    out = df[[data.ID_COL, data.TEXT_COL, data.LABEL_COL]].sort_values(data.ID_COL).reset_index(drop=True)
    out["group_sha256"] = out[data.TEXT_COL].map(lambda t: hashlib.sha256(duplicate_key(t).encode("utf-8")).hexdigest())
    groups = out.groupby("group_sha256")
    n_labels = groups[data.LABEL_COL].transform("nunique")
    first_row = groups[data.ID_COL].transform("min")
    reason = pd.Series("", index=out.index)
    is_copy = out[data.ID_COL] != first_row
    reason[is_copy] = "duplicate_of_" + first_row[is_copy].astype(str)
    reason[n_labels > 1] = "conflicting_labels"
    out["exclusion_reason"] = reason

    kept = out[out["exclusion_reason"] == ""]
    train_val, test = train_test_split(kept, test_size=TEST_SIZE, stratify=kept[data.LABEL_COL], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), stratify=train_val[data.LABEL_COL], random_state=seed
    )
    split = pd.Series(EXCLUDED, index=out.index)
    split[train.index] = "train"
    split[val.index] = "validation"
    split[test.index] = "test"
    return pd.DataFrame(
        {
            data.ID_COL: out[data.ID_COL],
            "text_sha256": out[data.TEXT_COL].map(data.text_sha256),
            "group_sha256": out["group_sha256"],
            data.LABEL_COL: out[data.LABEL_COL],
            "split": split,
            "exclusion_reason": out["exclusion_reason"],
        }
    )


def compare_manifests(expected: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Построчное сравнение двух манифестов. Возвращает отчёт с числом расхождений по колонкам."""
    cols = list(expected.columns)
    e = expected[cols].sort_values(data.ID_COL).reset_index(drop=True)
    a = actual[cols].sort_values(data.ID_COL).reset_index(drop=True)
    if len(e) != len(a) or not (e[data.ID_COL].values == a[data.ID_COL].values).all():
        return {"identical": False, "reason": "разный набор source_row", "rows_expected": len(e), "rows_actual": len(a)}
    mismatches = {c: int((e[c].astype(str).values != a[c].astype(str).values).sum()) for c in cols}
    return {"identical": all(v == 0 for v in mismatches.values()), "rows": len(e), "mismatches_by_column": mismatches}


def split_overlaps(manifest: pd.DataFrame) -> dict:
    """Число общих текстов (по SHA-256 текста и по ключу дедупликации) для каждой пары разбиений."""
    used = manifest[manifest["split"].isin(SPLITS)]
    result = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            for col in ("text_sha256", "group_sha256"):
                common = set(used.loc[used.split == a, col]) & set(used.loc[used.split == b, col])
                result[f"{a}-{b}:{col}"] = len(common)
    return result
