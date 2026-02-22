"""Evaluation metrics for anomaly detection.

Adapted from MSc thesis runner.py. Focuses on metrics suitable
for extreme class imbalance (insider threat / agent failure
scenarios where anomalies < 1%).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class DetectionMetrics:
    """Container for anomaly detection evaluation metrics."""

    auc_roc: float = 0.0
    auc_pr: float = 0.0
    recall_at_5fpr: float = 0.0
    recall_at_10fpr: float = 0.0
    precision_at_5fpr: float = 0.0
    precision_at_10fpr: float = 0.0
    threshold_at_5fpr: float = 0.0
    threshold_at_10fpr: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "recall@5%FPR": self.recall_at_5fpr,
            "recall@10%FPR": self.recall_at_10fpr,
            "precision@5%FPR": self.precision_at_5fpr,
            "precision@10%FPR": self.precision_at_10fpr,
        }


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> DetectionMetrics:
    """Compute detection metrics.

    Args:
        y_true: Binary ground truth (0=normal, 1=anomalous).
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        DetectionMetrics with all computed values.
    """
    metrics = DetectionMetrics()

    if len(np.unique(y_true)) < 2:
        return metrics

    # AUC-ROC
    metrics.auc_roc = float(roc_auc_score(y_true, scores))

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    metrics.auc_pr = float(auc(recall, precision))

    # ROC curve for threshold-based metrics
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Recall @ 5% FPR
    idx_5 = np.searchsorted(fpr, 0.05, side="right") - 1
    idx_5 = max(0, min(idx_5, len(tpr) - 1))
    metrics.recall_at_5fpr = float(tpr[idx_5])
    if idx_5 < len(thresholds):
        metrics.threshold_at_5fpr = float(thresholds[idx_5])

    # Recall @ 10% FPR
    idx_10 = np.searchsorted(fpr, 0.10, side="right") - 1
    idx_10 = max(0, min(idx_10, len(tpr) - 1))
    metrics.recall_at_10fpr = float(tpr[idx_10])
    if idx_10 < len(thresholds):
        metrics.threshold_at_10fpr = float(thresholds[idx_10])

    # Precision at those thresholds
    for fpr_target, attr_prefix in [
        (0.05, "precision_at_5fpr"),
        (0.10, "precision_at_10fpr"),
    ]:
        idx = np.searchsorted(fpr, fpr_target, side="right") - 1
        idx = max(0, min(idx, len(thresholds) - 1))
        if idx < len(thresholds):
            preds = (scores >= thresholds[idx]).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            prec = tp / max(tp + fp, 1)
            setattr(metrics, attr_prefix, float(prec))

    return metrics


def run_multi_seed(
    model_class,
    model_kwargs: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run experiment across multiple seeds.

    Returns mean and std of all metrics.
    """
    seeds = seeds or [42, 43, 44, 45, 46]
    all_metrics = []

    for seed in seeds:
        kwargs = {**model_kwargs, "seed": seed}
        model = model_class(**kwargs)
        model.fit(X_train)
        scores = model.score(X_test)
        m = compute_metrics(y_test, scores)
        all_metrics.append(m.to_dict())

    # Aggregate
    result = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    return result
