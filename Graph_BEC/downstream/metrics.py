"""Classification metrics and threshold utilities for downstream BEC probes."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_youden_threshold(labels, probabilities, fallback=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.unique(labels).size < 2:
        return float(fallback)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float(fallback)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    finite_thresholds = thresholds[finite]
    return float(finite_thresholds[np.argmax(scores)])


def classification_metrics(labels, probabilities, threshold=0.5):
    """Return fractions in [0, 1]; formatting as percentages happens in the report."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    true_positive = np.sum((labels == 1) & (predictions == 1))
    true_negative = np.sum((labels == 0) & (predictions == 0))
    false_negative = np.sum((labels == 1) & (predictions == 0))
    false_positive = np.sum((labels == 0) & (predictions == 1))
    total = len(labels)
    return {
        "ACC": float((true_positive + true_negative) / total) if total else 0.0,
        "SEN": float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0,
        "SPE": float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else 0.0,
        "AUC": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan"),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
    }


__all__ = ["classification_metrics", "select_youden_threshold"]
