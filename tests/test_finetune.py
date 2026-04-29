import numpy as np
import torch

from rubert_clf import model_source
from rubert_clf.dataset import ToxicCommentsDataset, make_dataloader, make_length_sorted_dataloader
from rubert_clf.finetune import FinetuneConfig, build_model, make_optimizer, predict_proba, train_epoch
from rubert_clf.runtime import get_device, set_seed

TEXTS = ["Ты дурак!", "Спасибо, очень полезно.", "Иди отсюда, урод", "Хорошего дня всем", "Заткнись", "Интересная статья"]
LABELS = [1, 0, 1, 0, 1, 0]
CPU = torch.device("cpu")


def _setup(tokenizer):
    set_seed(0)
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=32)
    return ds, build_model(model_source.MODEL_DIR)


def test_cpu_device_selected_without_gpu():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_classifier_head_is_new(tokenizer):
    # голова не из чекпойнта: две модели с разным seed дают разные веса головы и одинаковый энкодер
    set_seed(1)
    a = build_model(model_source.MODEL_DIR)
    set_seed(2)
    b = build_model(model_source.MODEL_DIR)
    assert not torch.equal(a.classifier.weight, b.classifier.weight)
    assert torch.equal(a.bert.embeddings.word_embeddings.weight, b.bert.embeddings.word_embeddings.weight)


def test_training_updates_encoder_and_reduces_loss(tokenizer):
    ds, model = _setup(tokenizer)
    loader = make_dataloader(ds, tokenizer, batch_size=6, shuffle=True)
    before = model.bert.encoder.layer[0].attention.self.query.weight.clone()
    optimizer = make_optimizer(model, FinetuneConfig(lr=1e-4))
    losses = [train_epoch(model, loader, optimizer, CPU)["train_loss"] for _ in range(8)]
    assert losses[-1] < losses[0]
    assert not torch.equal(before, model.bert.encoder.layer[0].attention.self.query.weight)


def test_eval_mode_is_deterministic_train_mode_is_not(tokenizer):
    ds, model = _setup(tokenizer)
    loader, order = make_length_sorted_dataloader(ds, tokenizer, batch_size=4)
    p1, _ = predict_proba(model, loader, CPU, order)
    p2, _ = predict_proba(model, loader, CPU, order)
    np.testing.assert_array_equal(p1, p2)
    batch = next(iter(loader))
    model.train()
    with torch.no_grad():
        a = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        b = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert not torch.allclose(a, b)  # dropout


def test_predict_restores_original_order(tokenizer):
    ds, model = _setup(tokenizer)
    loader, order = make_length_sorted_dataloader(ds, tokenizer, batch_size=2)
    probs, loss = predict_proba(model, loader, CPU, order)
    single = []
    for text, label in zip(TEXTS, LABELS):
        one_loader, one_order = make_length_sorted_dataloader(ToxicCommentsDataset([text], [label], tokenizer, 32), tokenizer, 1)
        single.append(predict_proba(model, one_loader, CPU, one_order)[0][0])
    np.testing.assert_allclose(probs, single, atol=1e-5)
    assert loss is not None and loss > 0
