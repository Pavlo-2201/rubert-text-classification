"""Метрики и выбор порога. Основная метрика — F1 токсичного класса (метка 1), как в классике."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """scores — монотонные оценки токсичности (вероятность или decision_function)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "f1_toxic": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_toxic": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_toxic": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "predicted_toxic_share": float(np.mean(y_pred == 1)),
    }
    if scores is not None and len(np.unique(scores)) > 1:
        result["roc_auc"] = roc_auc_score(y_true, scores)
        result["average_precision"] = average_precision_score(y_true, scores)
    result = {k: round(float(v), 4) for k, v in result.items()}
    result["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return result


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Порог, максимизирующий F1 токсичного класса на переданных (не тестовых!) оценках."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def bootstrap_f1_ci(y_true, y_pred, n_resamples: int = 1000, seed: int = 42, level: float = 0.95) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        values.append(f1_score(y_true[idx], y_pred[idx], pos_label=1, zero_division=0))
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return {"level": level, "n_resamples": n_resamples, "seed": seed, "low": round(float(low), 4), "high": round(float(high), 4)}
