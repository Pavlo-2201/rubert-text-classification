"""Замороженный энкодер: извлечение эмбеддингов текстов без обучения весов BERT.

Энкодер переводится в eval() (dropout выключен), градиенты не считаются (torch.no_grad),
поэтому веса BERT не меняются и эмбеддинг текста фиксирован. Обучается только классификатор
поверх эмбеддингов.

Пулинг:
- cls  — скрытое состояние токена [CLS] последнего слоя;
- mean — среднее скрытых состояний по реальным токенам (attention_mask=1), padding не учитывается.
"""

from __future__ import annotations

import numpy as np
import torch

from rubert_clf.dataset import ToxicCommentsDataset, make_length_sorted_dataloader

POOLINGS = ("cls", "mean")


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError(f"Неизвестный пулинг: {pooling}")


@torch.no_grad()
def extract_embeddings(encoder, dataset: ToxicCommentsDataset, tokenizer, device, batch_size: int = 64) -> dict[str, np.ndarray]:
    """Возвращает {'cls': [N, H], 'mean': [N, H]} в исходном порядке текстов."""
    encoder.eval().to(device)
    loader, order = make_length_sorted_dataloader(dataset, tokenizer, batch_size)
    chunks = {p: [] for p in POOLINGS}
    for batch in loader:
        mask = batch["attention_mask"].to(device)
        hidden = encoder(input_ids=batch["input_ids"].to(device), attention_mask=mask).last_hidden_state
        for p in POOLINGS:
            chunks[p].append(pool(hidden, mask, p).float().cpu().numpy())
    result = {}
    for p in POOLINGS:
        sorted_emb = np.concatenate(chunks[p])
        emb = np.empty_like(sorted_emb)
        emb[order] = sorted_emb
        result[p] = emb
    return result
