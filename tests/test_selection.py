import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from rubert_clf import model_source
from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader
from rubert_clf.finetune import FinetuneConfig, build_model, class_weights, fit, predict_proba
from rubert_clf.metrics import best_f1_threshold
from rubert_clf.runtime import set_seed

CPU = torch.device("cpu")
TEXTS = ["Ты дурак!", "Спасибо, очень полезно.", "Иди отсюда, урод", "Хорошего дня всем", "Заткнись", "Интересная статья",
         "Какой же ты тупой", "Согласен с автором", "Пошёл вон", "Отличная идея", "Бред несёшь, идиот", "Спасибо за помощь"]
LABELS = [1, 0] * 6


def test_balanced_class_weights():
    assert class_weights([0, 0, 0, 1], "none") is None
    weights = class_weights([0, 0, 0, 1], "balanced")
    assert torch.allclose(weights, torch.tensor([4 / 6, 4 / 2]))


def test_best_f1_threshold_on_separable_scores():
    threshold, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9])
    assert threshold == 0.3 and f1 == 1.0


def test_best_checkpoint_is_saved_and_reloads(tokenizer, tmp_path):
    set_seed(0)
    model = build_model(model_source.MODEL_DIR)
    ds = ToxicCommentsDataset(TEXTS, LABELS, tokenizer, max_length=16)
    config = FinetuneConfig(epochs=3, batch_size=4, lr=1e-4, class_weight="balanced")
    history = fit(model, ds, ds, tokenizer, config, CPU, checkpoint_dir=tmp_path, log=lambda *_: None)
    selection = json.loads((tmp_path / "best" / "selection.json").read_text())
    tuned = [h["validation_tuned"]["f1_toxic"] for h in history]
    assert selection["epoch"] == history[int(np.argmax(tuned))]["epoch"]  # первая эпоха с максимумом
    assert selection["config"]["class_weight"] == "balanced"

    best = history[selection["epoch"] - 1]
    reloaded = AutoModelForSequenceClassification.from_pretrained(tmp_path / "best")
    loader, order = make_length_sorted_dataloader(ds, tokenizer, 8)
    probs, _ = predict_proba(reloaded, loader, CPU, order)
    np.testing.assert_allclose(probs, best["val_probs"], atol=1e-5)
