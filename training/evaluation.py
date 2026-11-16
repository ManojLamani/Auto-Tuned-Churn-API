"""Metrics, threshold selection and risk tiers."""
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

CV_SCORING = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def best_f1_threshold(y_true, proba):
    """Threshold that maximises F1 on the given (out-of-fold) predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return round(float(thresholds[int(np.argmax(f1))]), 2)


def risk_tiers(threshold):
    return {"HIGH": threshold, "MEDIUM": round(threshold / 2, 4), "LOW": 0.0}


def risk_tier(probability, tiers):
    if probability >= tiers["HIGH"]:
        return "HIGH"
    if probability >= tiers["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def holdout_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
