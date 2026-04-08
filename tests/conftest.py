import pytest

from rubert_clf import data


@pytest.fixture(scope="session")
def dataset():
    if not data.DATASET_FILE.exists():
        pytest.skip("данные не скачаны: python scripts/download_data.py")
    return data.load_dataset()
