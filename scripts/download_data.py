"""Воспроизводимая загрузка данных (тот же файл, что в russian-text-classification).

Скачивает зафиксированный по коммиту файл из s-nlp/rudetoxifier, проверяет SHA-256,
выделяет блок Kaggle-датасета (2ch + pikabu) и проверяет отпечаток содержимого.

    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request

from rubert_clf import data


def download(url: str, dest, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"Уже скачан: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Скачиваю {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="скачать заново, даже если файл есть")
    args = parser.parse_args()

    download(data.SOURCE_URL, data.SOURCE_FILE, force=args.force)
    actual = data.sha256_file(data.SOURCE_FILE)
    if actual != data.SOURCE_SHA256:
        raise SystemExit(f"SHA-256 не совпал: ожидался {data.SOURCE_SHA256}, получен {actual}")
    print(f"SHA-256 исходного файла совпадает: {actual}")

    subset = data.extract_subset(data.read_source_csv(data.SOURCE_FILE))
    counts = subset[data.LABEL_COL].value_counts().to_dict()
    if len(subset) != data.EXPECTED_ROWS or counts != data.EXPECTED_LABEL_COUNTS:
        raise SystemExit(f"Неожиданный состав блока: {len(subset)} строк, метки {counts}")
    fingerprint = data.dataset_fingerprint(subset)
    if fingerprint != data.EXPECTED_FINGERPRINT:
        raise SystemExit(f"Отпечаток содержимого не совпал: {fingerprint}")
    subset.to_csv(data.DATASET_FILE, index=False)
    print(f"Отпечаток содержимого совпадает: {fingerprint}")
    print(f"Сохранено {len(subset)} строк в {data.DATASET_FILE}")


if __name__ == "__main__":
    main()
