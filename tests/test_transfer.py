"""Tests for cross-domain transfer evaluation."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics
from src.evaluation.transfer_analysis import (
    TransferResult,
    evaluate_transfer,
    transfer_summary_table,
)
from src.evaluation.owasp_mapper import (
    OWASP_CATEGORIES,
    evaluate_owasp_detection,
    owasp_detection_matrix,
)
from src.models.isolation_forest import IsolationForestDetector


@pytest.fixture
def synthetic_transfer_data():
    """Two synthetic domains with shared structure."""
    rng = np.random.RandomState(42)
    dim = 20

    # Source domain (CERT-like)
    cert_normal = rng.randn(200, dim).astype(np.float32)
    cert_anomaly = rng.randn(20, dim).astype(np.float32) + 2.5
    cert_train = cert_normal[:150]
    cert_test = np.vstack([cert_normal[150:], cert_anomaly])
    cert_y = np.array([0] * 50 + [1] * 20)

    # Target domain (agent-like, different distribution)
    agent_normal = rng.randn(100, dim).astype(np.float32) * 1.2
    agent_anomaly = rng.randn(10, dim).astype(np.float32) + 2.0
    agent_test = np.vstack([agent_normal[:50], agent_anomaly])
    agent_y = np.array([0] * 50 + [1] * 10)

    return {
        "cert": {
            "X_train": cert_train,
            "X_test": cert_test,
            "y_test": cert_y,
        },
        "agent": {
            "X_train": agent_normal[:80],
            "X_test": agent_test,
            "y_test": agent_y,
        },
    }


class TestMetrics:
    def test_compute_metrics(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 90 + [1] * 10)
        scores = rng.randn(100)
        scores[90:] += 2  # Make anomalies higher

        m = compute_metrics(y, scores)
        assert 0 < m.auc_roc <= 1.0
        assert 0 < m.auc_pr <= 1.0

    def test_single_class_returns_zero(self):
        y = np.zeros(50)
        scores = np.random.randn(50)
        m = compute_metrics(y, scores)
        assert m.auc_roc == 0.0


class TestTransferEvaluation:
    def test_evaluate_transfer(self, synthetic_transfer_data):
        model = IsolationForestDetector(
            n_estimators=50, seed=42
        )
        cert = synthetic_transfer_data["cert"]
        agent = synthetic_transfer_data["agent"]

        result = evaluate_transfer(
            model=model,
            X_train_source=cert["X_train"],
            X_test_source=cert["X_test"],
            y_test_source=cert["y_test"],
            X_test_target=agent["X_test"],
            y_test_target=agent["y_test"],
        )

        assert result.source_metrics is not None
        assert result.transfer_metrics is not None
        assert result.source_domain == "cert"
        assert result.target_domain == "agent"
        assert isinstance(result.auc_roc_drop, float)

    def test_summary_table(self, synthetic_transfer_data):
        model = IsolationForestDetector(
            n_estimators=50, seed=42
        )
        cert = synthetic_transfer_data["cert"]
        agent = synthetic_transfer_data["agent"]

        result = evaluate_transfer(
            model=model,
            X_train_source=cert["X_train"],
            X_test_source=cert["X_test"],
            y_test_source=cert["y_test"],
            X_test_target=agent["X_test"],
            y_test_target=agent["y_test"],
        )

        table = transfer_summary_table([result])
        assert "IsolationForest" in table
        assert "cert -> agent" in table


class TestOWASPMapper:
    def test_evaluate_owasp_detection(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 20).astype(np.float32)
        X_test = np.vstack([
            rng.randn(50, 20).astype(np.float32),
            rng.randn(10, 20).astype(np.float32) + 3,
            rng.randn(10, 20).astype(np.float32) + 2,
        ])
        y_test = np.array([0] * 50 + [1] * 10 + [1] * 10)
        owasp_labels = (
            [""] * 50 + ["ASI01"] * 10 + ["ASI09"] * 10
        )

        model = IsolationForestDetector(
            n_estimators=50, seed=42
        )
        model.fit(X_train)

        result = evaluate_owasp_detection(
            model, X_test, y_test, owasp_labels
        )

        assert result.model_name == "IsolationForest"
        assert "ASI01" in result.category_metrics
        assert "ASI09" in result.category_metrics
        # Categories not in test data should be blind spots
        assert len(result.blind_spots) > 0

    def test_detection_matrix_shape(self):
        from src.evaluation.owasp_mapper import (
            OWASPDetectionResult,
        )

        results = [
            OWASPDetectionResult(
                model_name="test",
                category_metrics={
                    "ASI01": {"auc_roc": 0.8, "auc_pr": 0.5,
                              "recall@10%FPR": 0.6, "n_samples": 10},
                },
            )
        ]
        matrix = owasp_detection_matrix(results)
        assert matrix.shape == (1, 10)
        assert matrix[0, 0] == 0.8  # ASI01
        assert np.isnan(matrix[0, 1])  # ASI02 missing
