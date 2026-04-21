import numpy as np
import torch

from rubert_clf import data, splits
from rubert_clf.dataset import ToxicCommentsDataset
from rubert_clf.embeddings import extract_embeddings, pool
from rubert_clf.tfidf_baseline import build_classical_baseline, load_classical_selection

TEXTS = ["Ты дурак!", "Очень длинный и вежливый комментарий про погоду и выходные, спасибо всем.", "ок", "Иди отсюда"]


def test_mean_pooling_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(pool(hidden, mask, "mean"), torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(pool(hidden, mask, "cls"), torch.tensor([[1.0, 1.0]]))


def test_embeddings_restore_order_and_match_single_text(tokenizer, encoder):
    ds = ToxicCommentsDataset(TEXTS, None, tokenizer, max_length=64)
    batched = extract_embeddings(encoder, ds, tokenizer, torch.device("cpu"), batch_size=3)
    for i, text in enumerate(TEXTS):
        single = extract_embeddings(encoder, ToxicCommentsDataset([text], None, tokenizer, 64), tokenizer, torch.device("cpu"))
        for p in ("cls", "mean"):
            np.testing.assert_allclose(batched[p][i], single[p][0], atol=1e-5)


def test_frozen_extraction_does_not_change_weights(tokenizer, encoder):
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    extract_embeddings(encoder, ToxicCommentsDataset(TEXTS, None, tokenizer, 32), tokenizer, torch.device("cpu"))
    assert all(torch.equal(before[k], v) for k, v in encoder.state_dict().items())


def test_tfidf_baseline_uses_classical_config():
    selection = load_classical_selection()
    model = build_classical_baseline()
    assert model.threshold == selection["threshold"]["used"]
    assert model.response_method == "decision_function"
    assert model.estimator.named_steps["clf"].C == selection["best_config"]["model_params"]["C"]


def test_tfidf_baseline_reproduces_classical_validation(dataset):
    parts = splits.load_splits(dataset)
    train, val = parts["train"], parts["validation"]
    model = build_classical_baseline().fit(train[data.TEXT_COL], train[data.LABEL_COL])
    pred = model.predict(val[data.TEXT_COL])
    expected = load_classical_selection()["validation"]["used_threshold"]["confusion_matrix"]
    tp = int(((pred == 1) & (val[data.LABEL_COL] == 1)).sum())
    fp = int(((pred == 1) & (val[data.LABEL_COL] == 0)).sum())
    assert (tp, fp) == (expected["tp"], expected["fp"])
