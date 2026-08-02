import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def classification_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    return classification_metrics_from_predictions(
        labels,
        probabilities,
        predictions,
    )


def classification_metrics_from_predictions(labels, probabilities, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = roc_auc_score(labels, probabilities)
    return {
        "ACC": float(accuracy),
        "SEN": float(sensitivity),
        "SPE": float(specificity),
        "AUC": float(auc),
    }


def select_youden_threshold(labels, probabilities, fallback=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.unique(labels).size < 2:
        return float(fallback)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        probabilities,
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float(fallback)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    finite_thresholds = thresholds[finite]
    return float(finite_thresholds[np.argmax(scores)])
