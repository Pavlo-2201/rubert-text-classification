"""Загрузка обученной модели и предсказания для сырых текстов.

Папка модели (результат train.py) — формат Hugging Face плюс метаданные:
    config.json, model.safetensors, tokenizer.json, tokenizer_config.json, ... — веса и токенизатор;
    selection.json — выбранная эпоха, порог, метрики validation и конфигурация обучения.
Нормализация и max_length берутся из метаданных, поэтому на вход подаются сырые тексты.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader
from rubert_clf.finetune import predict_proba
from rubert_clf.runtime import get_device


class ToxicityClassifier:
    def __init__(self, model, tokenizer, selection: dict, device: torch.device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.selection = selection
        self.device = device
        self.threshold = float(selection["threshold"])
        self.max_length = int(selection["config"]["max_length"])

    @classmethod
    def load(cls, model_dir: Path, device: str = "auto") -> "ToxicityClassifier":
        model_dir = Path(model_dir)
        selection_path = model_dir / "selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(f"{selection_path} не найден. Обучите модель: python train.py")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return cls(model, tokenizer, selection, get_device(device))

    def predict_proba(self, texts, batch_size: int = 128) -> np.ndarray:
        """Вероятность токсичного класса (softmax), в порядке входных текстов."""
        texts = list(texts)
        if not texts:
            return np.empty(0)
        ds = ToxicCommentsDataset(texts, None, self.tokenizer, self.max_length)
        loader, order = make_length_sorted_dataloader(ds, self.tokenizer, batch_size)
        return predict_proba(self.model, loader, self.device, order)[0]

    def predict(self, texts, batch_size: int = 128) -> np.ndarray:
        return (self.predict_proba(texts, batch_size) >= self.threshold).astype(int)
