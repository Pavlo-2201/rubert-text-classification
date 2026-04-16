"""Минимальная нормализация, как в классическом проекте: NFC, схлопывание пробелов, обрезка.

Регистр, пунктуацию и эмоджи не трогаем: токенизатор rubert-tiny2 регистрозависимый
(do_lower_case=false), а «!!!», капслок и маскировка мата могут нести сигнал.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalize_texts(texts) -> list[str]:
    return [normalize_text(t) for t in texts]
