"""Классический baseline: char_wb TF-IDF + LinearSVC с порогом на decision score.

Конфигурация и порог берутся из reference/classical/best_model.json — файла, выбранного
кросс-валидацией на train в проекте russian-text-classification. Здесь модель обучается
заново на том же train, чтобы все три подхода измерялись одним кодом и на одной машине
(в том числе время обучения и инференса). Параметры TfidfVectorizer и LinearSVC по умолчанию
повторяют классический проект (min_df=2, sublinear_tf, max_iter=5000, random_state=42).
"""

from __future__ import annotations

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from rubert_clf import data
from rubert_clf.text import normalize_texts

CLASSICAL_DIR = data.PROJECT_ROOT / "reference" / "classical"
BEST_MODEL_PATH = CLASSICAL_DIR / "best_model.json"
SEED = 42


def load_classical_selection() -> dict:
    return json.loads(BEST_MODEL_PATH.read_text(encoding="utf-8"))


def build_tfidf_svc(config: dict) -> Pipeline:
    if config["model"] != "linearsvc" or set(config) - {"name", "model", "model_params", "char"}:
        raise ValueError(f"Поддерживается только char TF-IDF + LinearSVC, получено: {config}")
    char = {"analyzer": "char_wb", "lowercase": True, "ngram_range": (2, 5), "min_df": 2, "sublinear_tf": True, **config["char"]}
    char["ngram_range"] = tuple(char["ngram_range"])
    clf = LinearSVC(**{"C": 1.0, "max_iter": 5000, "random_state": SEED, **config["model_params"]})
    return Pipeline([("normalize", FunctionTransformer(normalize_texts)), ("tfidf", TfidfVectorizer(**char)), ("clf", clf)])


def build_classical_baseline() -> FixedThresholdClassifier:
    """Pipeline с порогом на decision_function (не вероятность)."""
    selection = load_classical_selection()
    return FixedThresholdClassifier(
        build_tfidf_svc(selection["best_config"]),
        threshold=selection["threshold"]["used"],
        response_method="decision_function",
    )
