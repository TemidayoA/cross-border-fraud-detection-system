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
