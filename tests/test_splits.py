import pandas as pd
import pytest

from rubert_clf import data, splits


def test_manifest_sha_is_pinned():
    manifest = splits.load_manifest()  # бросит исключение при несовпадении SHA-256
    assert len(manifest) == data.EXPECTED_ROWS


def test_no_text_shared_between_splits():
    overlaps = splits.split_overlaps(splits.load_manifest())
    assert all(v == 0 for v in overlaps.values()), overlaps


def test_split_sizes_match_classical_fingerprint():
    manifest = splits.load_manifest()
    counts = splits.load_fingerprint()["counts"]
    for name in (*splits.SPLITS, splits.EXCLUDED):
        assert (manifest.split == name).sum() == counts[name]["rows"]


def test_dataset_fingerprint(dataset):
    assert data.dataset_fingerprint(dataset) == data.EXPECTED_FINGERPRINT


def test_rebuilt_manifest_identical(dataset):
    report = splits.compare_manifests(splits.load_manifest(), splits.rebuild_manifest(dataset))
    assert report["identical"], report


def test_load_splits_detects_changed_text(dataset):
    changed = dataset.copy()
    changed.loc[0, data.TEXT_COL] = changed.loc[0, data.TEXT_COL] + "!"
    with pytest.raises(ValueError):
        splits.load_splits(changed)


def test_compare_manifests_detects_moved_row():
    manifest = splits.load_manifest()
    moved = manifest.copy()
    idx = moved.index[moved.split == "test"][0]
    moved.loc[idx, "split"] = "train"
    report = splits.compare_manifests(manifest, moved)
    assert not report["identical"] and report["mismatches_by_column"]["split"] == 1


def test_overlap_detection():
    toy = pd.DataFrame(
        {"split": ["train", "test", "validation"], "text_sha256": ["a", "a", "b"], "group_sha256": ["g1", "g2", "g3"]}
    )
    assert splits.split_overlaps(toy)["train-test:text_sha256"] == 1
