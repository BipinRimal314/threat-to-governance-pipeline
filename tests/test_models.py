"""Tests for anomaly detection models."""

import numpy as np
import pytest

from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.deep_clustering import DeepClusteringDetector


@pytest.fixture
def synthetic_static_data():
    """Generate synthetic static data for testing."""
    rng = np.random.RandomState(42)
    X_normal = rng.randn(200, 20).astype(np.float32)
    X_anomaly = rng.randn(20, 20).astype(np.float32) + 3

    X_train = X_normal[:150]
    X_test = np.vstack([X_normal[150:], X_anomaly])
    y_test = np.array([0] * 50 + [1] * 20, dtype=np.int32)

    return X_train, X_test, y_test


@pytest.fixture
def synthetic_temporal_data():
    """Generate synthetic temporal data for testing."""
    rng = np.random.RandomState(42)
    seq_len = 7
    n_features = 20

    X_normal = rng.randn(100, seq_len, n_features).astype(
        np.float32
    )
    X_anomaly = (
        rng.randn(10, seq_len, n_features).astype(np.float32) + 3
    )

    X_train = X_normal[:80]
    X_test = np.concatenate([X_normal[80:], X_anomaly])
    y_test = np.array([0] * 20 + [1] * 10, dtype=np.int32)

    return X_train, X_test, y_test


class TestIsolationForest:
    def test_fit_score(self, synthetic_static_data):
        X_train, X_test, y_test = synthetic_static_data
        model = IsolationForestDetector(
            n_estimators=50, seed=42
        )
        model.fit(X_train)
        scores = model.score(X_test)

        assert scores.shape == (70,)
        assert model.is_fitted
        assert model.threshold_ is not None

    def test_predict(self, synthetic_static_data):
        X_train, X_test, _ = synthetic_static_data
        model = IsolationForestDetector(seed=42)
        model.fit(X_train)
        preds = model.predict(X_test)

        assert preds.shape == (70,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomalies_score_higher(self, synthetic_static_data):
        X_train, X_test, y_test = synthetic_static_data
        model = IsolationForestDetector(seed=42)
        model.fit(X_train)
        scores = model.score(X_test)

        normal_mean = scores[y_test == 0].mean()
        anomaly_mean = scores[y_test == 1].mean()
        assert anomaly_mean > normal_mean

    def test_fit_predict_result(self, synthetic_static_data):
        X_train, X_test, _ = synthetic_static_data
        model = IsolationForestDetector(seed=42)
        result = model.fit_predict(X_train, X_test)

        assert result.scores.shape == (70,)
        assert result.train_time > 0
        assert result.inference_time > 0

    def test_score_before_fit_raises(self):
        model = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            model.score(np.zeros((10, 20)))


class TestLSTMAutoencoder:
    def test_fit_score(self, synthetic_temporal_data):
        X_train, X_test, y_test = synthetic_temporal_data
        model = LSTMAutoencoderDetector(
            encoder_units=[16, 8],
            decoder_units=[8, 16],
            latent_dim=4,
            epochs=5,
            seed=42,
        )
        model.fit(X_train)
        scores = model.score(X_test)

        assert scores.shape == (30,)
        assert model.is_fitted

    def test_encode(self, synthetic_temporal_data):
        X_train, X_test, _ = synthetic_temporal_data
        model = LSTMAutoencoderDetector(
            encoder_units=[16, 8],
            decoder_units=[8, 16],
            latent_dim=4,
            epochs=5,
            seed=42,
        )
        model.fit(X_train)
        latent = model.encode(X_test)

        assert latent.shape == (30, 4)

    def test_score_per_timestep(self, synthetic_temporal_data):
        X_train, X_test, _ = synthetic_temporal_data
        model = LSTMAutoencoderDetector(
            encoder_units=[16, 8],
            decoder_units=[8, 16],
            latent_dim=4,
            epochs=5,
            seed=42,
        )
        model.fit(X_train)
        ts_scores = model.score_per_timestep(X_test)

        assert ts_scores.shape == (30, 7)


class TestDeepClustering:
    def test_fit_score(self, synthetic_static_data):
        X_train, X_test, y_test = synthetic_static_data
        model = DeepClusteringDetector(
            encoder_layers=[32, 16],
            latent_dim=8,
            n_clusters=3,
            pretrain_epochs=5,
            seed=42,
        )
        model.fit(X_train)
        scores = model.score(X_test)

        assert scores.shape == (70,)
        assert model.is_fitted

    def test_encode(self, synthetic_static_data):
        X_train, X_test, _ = synthetic_static_data
        model = DeepClusteringDetector(
            encoder_layers=[32, 16],
            latent_dim=8,
            n_clusters=3,
            pretrain_epochs=5,
            seed=42,
        )
        model.fit(X_train)
        latent = model.encode(X_test)

        assert latent.shape == (70, 8)

    def test_cluster_labels(self, synthetic_static_data):
        X_train, X_test, _ = synthetic_static_data
        model = DeepClusteringDetector(
            encoder_layers=[32, 16],
            latent_dim=8,
            n_clusters=3,
            pretrain_epochs=5,
            seed=42,
        )
        model.fit(X_train)
        labels = model.get_cluster_labels(X_test)

        assert labels.shape == (70,)
        assert set(np.unique(labels)).issubset({0, 1, 2})
