"""Дообучение (fine-tuning) RuBERT для классификации последовательностей.

В отличие от замороженного encoder, здесь обучаются все веса: и энкодер, и линейная голова
поверх [CLS] (BertForSequenceClassification: pooler dense+tanh -> dropout -> Linear(312, 2)).
Голова в чекпойнте отсутствует и инициализируется случайно, поэтому модель становится
классификатором только после обучения.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


@dataclass
class FinetuneConfig:
    max_length: int = 128
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def build_model(model_dir) -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)


def make_optimizer(model: nn.Module, config: FinetuneConfig) -> torch.optim.Optimizer:
    # weight decay не применяется к bias и LayerNorm — стандартная практика для BERT
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm" in name else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr,
    )


def train_epoch(model, loader, optimizer, device, loss_fn=None) -> dict:
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.train()  # включает dropout
    total, n = 0.0, 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = loss_fn(logits, batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch["labels"])
        n += len(batch["labels"])
    return {"train_loss": total / n, "seconds": time.perf_counter() - start}


@torch.no_grad()
def predict_proba(model, loader, device, order=None) -> tuple[np.ndarray, float | None]:
    """Вероятность токсичного класса (softmax по логитам) и средний loss, если есть метки.

    order — перестановка из make_length_sorted_dataloader; выход возвращается в исходный порядок.
    """
    model.eval()  # выключает dropout
    probs, total, n = [], 0.0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        if "labels" in batch:
            total += loss_fn(logits, batch["labels"]).item()
            n += len(batch["labels"])
    probs = np.concatenate(probs)
    if order is not None:
        restored = np.empty_like(probs)
        restored[order] = probs
        probs = restored
    return probs, (total / n if n else None)
