"""Cross-domain transfer evaluation.

Measures how well models trained on one domain (e.g., CMU-CERT)
perform when applied to the other domain (e.g., agent traces)
through the UBFS representation.

This is the core experimental contribution of the project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .metrics import DetectionMetrics, compute_metrics


@dataclass
class TransferResult:
    """Results from a cross-domain transfer experiment."""

    source_domain: str
    target_domain: str
    model_name: str

    # Within-domain performance (baseline)
    source_metrics: Optional[DetectionMetrics] = None

    # Cross-domain performance (the transfer result)
    transfer_metrics: Optional[DetectionMetrics] = None

    # Performance drop
    auc_roc_drop: float = 0.0
    auc_pr_drop: float = 0.0

    def compute_drop(self):
        """Compute performance degradation from transfer."""
        if self.source_metrics and self.transfer_metrics:
            self.auc_roc_drop = (
                self.source_metrics.auc_roc
                - self.transfer_metrics.auc_roc
            )
            self.auc_pr_drop = (
                self.source_metrics.auc_pr
                - self.transfer_metrics.auc_pr
            )


def evaluate_transfer(
    model,
    X_train_source: np.ndarray,
    X_test_source: np.ndarray,
    y_test_source: np.ndarray,
    X_test_target: np.ndarray,
    y_test_target: np.ndarray,
    source_domain: str = "cert",
    target_domain: str = "agent",
) -> TransferResult:
    """Evaluate cross-domain transfer for a single model.

    Protocol:
        1. Train model on source domain training data.
        2. Evaluate on source domain test data (baseline).
        3. Evaluate same model on target domain test data
           (transfer).
        4. Compare performance.

    Args:
        model: Anomaly detector implementing fit() and score().
        X_train_source: Training data from source domain.
        X_test_source: Test data from source domain.
        y_test_source: Labels for source test data.
        X_test_target: Test data from target domain.
        y_test_target: Labels for target test data.
        source_domain: Name of source domain.
        target_domain: Name of target domain.

    Returns:
        TransferResult with baseline and transfer metrics.
    """
    # Train on source domain
    model.fit(X_train_source)

    # Evaluate on source (baseline)
    source_scores = model.score(X_test_source)
    source_metrics = compute_metrics(y_test_source, source_scores)

    # Evaluate on target (transfer)
    target_scores = model.score(X_test_target)
    transfer_metrics = compute_metrics(
        y_test_target, target_scores
    )

    result = TransferResult(
        source_domain=source_domain,
        target_domain=target_domain,
        model_name=model.name,
        source_metrics=source_metrics,
        transfer_metrics=transfer_metrics,
    )
    result.compute_drop()

    return result


def evaluate_all_transfers(
    models: Dict[str, object],
    cert_data: dict,
    agent_data: dict,
) -> List[TransferResult]:
    """Run transfer experiments for all models both directions.

    Args:
        models: Dict of model_name -> model_instance.
        cert_data: Dict with X_train, X_test, y_test for CERT.
        agent_data: Dict with X_train, X_test, y_test for agents.

    Returns:
        List of TransferResult for all model/direction combos.
    """
    results = []

    for name, model in models.items():
        # CERT → Agent
        r1 = evaluate_transfer(
            model=model,
            X_train_source=cert_data["X_train"],
            X_test_source=cert_data["X_test"],
            y_test_source=cert_data["y_test"],
            X_test_target=agent_data["X_test"],
            y_test_target=agent_data["y_test"],
            source_domain="cert",
            target_domain="agent",
        )
        results.append(r1)

        # Recreate model for reverse direction
        model_class = type(model)
        model_reverse = model_class(**{
            k: v for k, v in model.get_params().items()
            if k != "name"
        })

        # Agent → CERT
        r2 = evaluate_transfer(
            model=model_reverse,
            X_train_source=agent_data["X_train"],
            X_test_source=agent_data["X_test"],
            y_test_source=agent_data["y_test"],
            X_test_target=cert_data["X_test"],
            y_test_target=cert_data["y_test"],
            source_domain="agent",
            target_domain="cert",
        )
        results.append(r2)

    return results


def transfer_summary_table(
    results: List[TransferResult],
) -> str:
    """Format transfer results as a markdown table."""
    lines = [
        "| Model | Direction | Source AUC-ROC | Transfer AUC-ROC | Drop |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        src_auc = (
            f"{r.source_metrics.auc_roc:.3f}"
            if r.source_metrics else "N/A"
        )
        tgt_auc = (
            f"{r.transfer_metrics.auc_roc:.3f}"
            if r.transfer_metrics else "N/A"
        )
        drop = f"{r.auc_roc_drop:+.3f}"
        direction = f"{r.source_domain} -> {r.target_domain}"
        lines.append(
            f"| {r.model_name} | {direction} | "
            f"{src_auc} | {tgt_auc} | {drop} |"
        )

    return "\n".join(lines)
