"""Токенизация, torch Dataset и DataLoader с динамическим padding.

- Тексты токенизируются один раз при создании Dataset, с truncation до max_length
  (включая [CLS] и [SEP]); padding на этом шаге не делается.
- Collate дополняет батч [PAD] до длины самого длинного текста в батче и строит
  attention_mask: 1 — реальный токен, 0 — padding. Так батч из коротких комментариев
  не растягивается до max_length, и вычисления на padding не тратятся.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from rubert_clf.text import normalize_texts


class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = normalize_texts(texts)
        self.labels = None if labels is None else [int(y) for y in labels]
        self.max_length = max_length
        # Длины без обрезки нужны, чтобы знать, сколько текстов теряют хвост при truncation.
        self.full_lengths = [len(ids) for ids in tokenizer(self.texts, truncation=False)["input_ids"]]
        encoded = tokenizer(self.texts, truncation=True, max_length=max_length, padding=False)
        self.input_ids = encoded["input_ids"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {"input_ids": self.input_ids[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    @property
    def truncated_share(self) -> float:
        return float(np.mean([n > self.max_length for n in self.full_lengths]))


class PaddingCollator:
    """Дополняет батч справа pad_token_id и строит attention_mask."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict]) -> dict:
        width = max(len(it["input_ids"]) for it in items)
        input_ids = torch.full((len(items), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), width), dtype=torch.long)
        for i, it in enumerate(items):
            n = len(it["input_ids"])
            input_ids[i, :n] = torch.tensor(it["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "labels" in items[0]:
            batch["labels"] = torch.tensor([it["labels"] for it in items], dtype=torch.long)
        return batch


def make_dataloader(dataset: ToxicCommentsDataset, tokenizer, batch_size: int, shuffle: bool, seed: int = 42) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=PaddingCollator(tokenizer.pad_token_id),
    )


def make_length_sorted_dataloader(dataset: ToxicCommentsDataset, tokenizer, batch_size: int) -> tuple[DataLoader, np.ndarray]:
    """Для инференса: батчи из текстов близкой длины, меньше padding.

    Возвращает DataLoader и порядок индексов; выход модели нужно вернуть в исходный порядок:
    result[order] = outputs.
    """
    order = np.argsort([len(ids) for ids in dataset.input_ids], kind="stable")
    loader = DataLoader(Subset(dataset, order.tolist()), batch_size=batch_size, shuffle=False,
                        collate_fn=PaddingCollator(tokenizer.pad_token_id))
    return loader, order
