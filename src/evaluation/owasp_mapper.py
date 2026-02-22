"""Map anomaly detections to OWASP Agentic Application risks.

Correlates model predictions with OWASP ASI01-ASI10 categories
to evaluate which risks each model can and cannot detect.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_metrics

# OWASP Top 10 for Agentic Applications (December 2025)
OWASP_CATEGORIES = {
    "ASI01": "Agent Goal Hijack",
    "ASI02": "Tool Misuse",
    "ASI03": "Identity & Privilege Abuse",
    "ASI04": "Cascading Hallucinations",
    "ASI05": "Memory Poisoning",
    "ASI06": "Supply Chain Compromise",
    "ASI07": "Unsafe Code Generation",
    "ASI08": "Inadequate Sandboxing",
    "ASI09": "Excessive Agency",
    "ASI10": "Rogue Agents",
}

# Mapping: which insider threat models are theoretically
# suited for which OWASP risks
MODEL_CATEGORY_AFFINITY = {
    "IsolationForest": {
        "ASI02": 0.8,   # Feature combinations
        "ASI03": 0.7,   # Access pattern anomalies
        "ASI09": 0.9,   # Action space breadth
        "ASI10": 0.6,   # General anomaly
    },
    "LSTMAutoencoder": {
        "ASI01": 0.9,   # Sequence deviation
        "ASI04": 0.7,   # Reconstruction error on drift
        "ASI05": 0.8,   # Injected temporal anomalies
        "ASI07": 0.6,   # Sequence-level scoring
        "ASI10": 0.7,   # Persistent deviation
    },
    "DeepClustering": {
        "ASI03": 0.8,   # Divergent access clusters
        "ASI06": 0.7,   # Out-of-distribution
        "ASI08": 0.6,   # Boundary clusters
        "ASI10": 0.8,   # Behavioural profiling
    },
}


@dataclass
class OWASPDetectionResult:
    """Detection results mapped to OWASP categories."""

    model_name: str
    # Per-category metrics
    category_metrics: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    # Categories with no detection capability
    blind_spots: List[str] = field(default_factory=list)


def evaluate_owasp_detection(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    owasp_labels: List[str],
) -> OWASPDetectionResult:
    """Evaluate model detection rate per OWASP category.

    Args:
        model: Fitted anomaly detector.
        X_test: Test features.
        y_test: Binary labels.
        owasp_labels: OWASP category for each test sample
            (empty string for normal samples).

    Returns:
        Detection results broken down by OWASP category.
    """
    scores = model.score(X_test)
    result = OWASPDetectionResult(model_name=model.name)

    for cat_code in OWASP_CATEGORIES:
        # Select samples for this category
        mask = np.array([
            label == cat_code for label in owasp_labels
        ])
        n_positive = mask.sum()

        if n_positive == 0:
            result.blind_spots.append(cat_code)
            continue

        # Create binary labels: this category vs normal
        cat_y = np.zeros_like(y_test)
        cat_y[mask] = 1

        # Also include some normal samples for metric computation
        normal_mask = np.array([label == "" for label in owasp_labels])
        eval_mask = mask | normal_mask

        if eval_mask.sum() < 2 or cat_y[eval_mask].sum() == 0:
            result.blind_spots.append(cat_code)
            continue

        metrics = compute_metrics(
            cat_y[eval_mask], scores[eval_mask]
        )
        result.category_metrics[cat_code] = {
            "auc_roc": metrics.auc_roc,
            "auc_pr": metrics.auc_pr,
            "recall@10%FPR": metrics.recall_at_10fpr,
            "n_samples": int(n_positive),
        }

    return result


def owasp_detection_matrix(
    results: List[OWASPDetectionResult],
) -> np.ndarray:
    """Create detection rate matrix (models x categories).

    Returns:
        Matrix of shape (n_models, 10) with AUC-ROC values.
        NaN for undetectable categories.
    """
    categories = list(OWASP_CATEGORIES.keys())
    matrix = np.full(
        (len(results), len(categories)), np.nan
    )

    for i, result in enumerate(results):
        for j, cat in enumerate(categories):
            if cat in result.category_metrics:
                matrix[i, j] = result.category_metrics[cat][
                    "auc_roc"
                ]

    return matrix


def owasp_summary_table(
    results: List[OWASPDetectionResult],
) -> str:
    """Format OWASP detection results as markdown table."""
    categories = list(OWASP_CATEGORIES.keys())
    header = "| Model | " + " | ".join(categories) + " |"
    sep = "|---|" + "|".join(["---"] * len(categories)) + "|"
    lines = [header, sep]

    for result in results:
        cells = [result.model_name]
        for cat in categories:
            if cat in result.category_metrics:
                val = result.category_metrics[cat]["auc_roc"]
                cells.append(f"{val:.2f}")
            else:
                cells.append("--")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
