"""Этап 1: проверка, что данные и разбиения полностью совпадают с классическим проектом.

Проверяется:
1. SHA-256 исходного файла и отпечаток содержимого (как в data_fingerprint.json классики);
2. SHA-256 манифеста в этом репозитории равен зафиксированному в классическом проекте;
3. манифест, заново построенный из данных по тому же правилу, совпадает построчно;
4. хеши текстов и метки из данных совпадают с манифестом (load_splits);
5. между разбиениями нет общих текстов ни по точному хешу, ни по ключу дедупликации;
6. если указан путь к классическому репозиторию — его манифест и fingerprint побайтно равны нашим.

    python scripts/check_splits.py [--classical-repo ../russian-text-classification]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rubert_clf import data, splits

REPORT = data.PROJECT_ROOT / "reports" / "split_check.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classical-repo", type=Path, help="рабочая копия russian-text-classification (необязательно)")
    args = parser.parse_args()

    fingerprint = splits.load_fingerprint()
    report: dict = {}

    report["source_file_sha256"] = data.sha256_file(data.SOURCE_FILE)
    report["source_file_sha256_ok"] = report["source_file_sha256"] == fingerprint["source"]["file_sha256"]

    dataset = data.load_dataset()
    report["content_fingerprint"] = data.dataset_fingerprint(dataset)
    report["content_fingerprint_ok"] = report["content_fingerprint"] == fingerprint["content_fingerprint"]

    manifest = splits.load_manifest()  # проверяет SHA-256 манифеста
    report["manifest_sha256"] = splits.MANIFEST_SHA256
    report["manifest_sha256_ok"] = splits.MANIFEST_SHA256 == fingerprint["manifest_sha256"]

    rebuilt = splits.rebuild_manifest(dataset)
    report["rebuilt_vs_manifest"] = splits.compare_manifests(manifest, rebuilt)

    parts = splits.load_splits(dataset, manifest)
    report["split_counts"] = {
        name: {"rows": len(df), "label_counts": {str(k): int(v) for k, v in df[data.LABEL_COL].value_counts().sort_index().items()}}
        for name, df in parts.items()
    }
    expected_counts = {name: fingerprint["counts"][name] for name in splits.SPLITS}
    report["split_counts_ok"] = report["split_counts"] == expected_counts
    report["overlaps"] = splits.split_overlaps(manifest)
    report["no_overlaps"] = all(v == 0 for v in report["overlaps"].values())

    if args.classical_repo:
        classical_dir = args.classical_repo / "data" / "splits"
        report["classical_repo"] = {
            "manifest_identical": data.sha256_file(classical_dir / "split_manifest.csv") == data.sha256_file(splits.MANIFEST_PATH),
            "fingerprint_identical": (classical_dir / "data_fingerprint.json").read_bytes() == splits.FINGERPRINT_PATH.read_bytes(),
        }

    checks = [
        report["source_file_sha256_ok"],
        report["content_fingerprint_ok"],
        report["manifest_sha256_ok"],
        report["rebuilt_vs_manifest"]["identical"],
        report["split_counts_ok"],
        report["no_overlaps"],
        *report.get("classical_repo", {}).values(),
    ]
    report["passed"] = bool(all(checks))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Проверка разбиений не пройдена")


if __name__ == "__main__":
    main()
