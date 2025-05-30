# src/evaluate.py

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def evaluate_binary_classification(
    y_true,
    y_pred,
    y_prob=None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["roc_auc"] = auc
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics
def print_classification_report(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))


def print_confusion(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def evaluate_and_print(
    task_name: str,
    y_true,
    y_pred,
    y_prob=None,
):
    print(f"\n=== Evaluation for {task_name} ===")
    metrics = evaluate_binary_classification(y_true, y_pred, y_prob)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print_classification_report(y_true, y_pred)
    print_confusion(y_true, y_pred)
    return metrics