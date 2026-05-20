import numpy as np
import pandas as pd

from rubert_clf import data, splits
from rubert_clf.errors import categorize, oov_share, train_vocabulary
from rubert_clf.metrics import mcnemar, paired_bootstrap_f1_diff
from rubert_clf.tfidf_baseline import CLASSICAL_DIR


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 1, 0, 1, 0])  # верно: 0, 1, 2
    b = np.array([0, 1, 0, 0, 0])  # верно: 1, 2, 3
    result = mcnemar(y, a, b)
    assert (result["only_first_correct"], result["only_second_correct"]) == (1, 1)
    assert result["p_value"] == 1.0


def test_paired_bootstrap_is_zero_for_identical_predictions():
    y = np.array([0, 1, 1, 0, 1, 0])
    result = paired_bootstrap_f1_diff(y, y, y, n=50)
    assert result == {"mean": 0.0, "low": 0.0, "high": 0.0}


def test_error_groups_match_classical_project(dataset):
    """Группы ошибок считаются теми же правилами, что в классике: сверка с её validation_errors.csv."""
    parts = splits.load_splits(dataset)
    vocab = train_vocabulary(parts["train"][data.TEXT_COL])
    classical = pd.read_csv(CLASSICAL_DIR / "validation_errors.csv", keep_default_na=False)
    texts = parts["validation"].set_index(data.ID_COL).loc[classical[data.ID_COL], data.TEXT_COL]
    for (_, row), text in zip(classical.iterrows(), texts):
        share = oov_share(text, vocab)
        assert round(share, 3) == row["oov_share"]
        for group, value in categorize(text, share).items():
            assert str(value) == str(row[group]), (row[data.ID_COL], group)
