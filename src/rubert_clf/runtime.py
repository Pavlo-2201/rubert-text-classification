"""Выбор устройства и фиксация seed."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен cuda, но GPU недоступен")
    return torch.device(preference)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
