import pytest

from rubert_clf import data


@pytest.fixture(scope="session")
def dataset():
    if not data.DATASET_FILE.exists():
        pytest.skip("данные не скачаны: python scripts/download_data.py")
    return data.load_dataset()


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    from rubert_clf import model_source

    if not (model_source.MODEL_DIR / "config.json").exists():
        pytest.skip("модель не скачана: python scripts/download_model.py")
    return AutoTokenizer.from_pretrained(model_source.MODEL_DIR)


@pytest.fixture(scope="session")
def encoder(tokenizer):
    from transformers import AutoModel

    from rubert_clf import model_source

    return AutoModel.from_pretrained(model_source.MODEL_DIR).eval()
