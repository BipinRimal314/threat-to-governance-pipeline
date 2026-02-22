"""Base interface for cross-domain anomaly detection models.

All models implement a common interface so that the same detector
can operate on UBFS vectors regardless of whether the source is
CMU-CERT employee data or AI agent traces.

Adapted from MSc thesis (Thesis_work/src/models/base.py),
with PyTorch backend replacing TensorFlow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModelResult:
    """Container for model prediction results."""

    scores: np.ndarray
    predictions: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    train_time: float = 0.0
    inference_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection models.

    Convention: higher score = more anomalous.
    Threshold defaults to 95th percentile of training scores.
    """

    def __init__(self, name: str, seed: int = 42, **kwargs):
        self.name = name
        self.seed = seed
        self.is_fitted = False
        self.threshold_: Optional[float] = None
        self._set_seed()

    def _set_seed(self) -> None:
        np.random.seed(self.seed)

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "BaseAnomalyDetector":
        """Train on normal data. y is ignored for unsupervised."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores (n_samples,). Higher = more anomalous."""
        pass

    def predict(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Binary predictions. 1 = anomaly, 0 = normal."""
        scores = self.score(X)
        if threshold is None:
            threshold = getattr(
                self, "threshold_", np.percentile(scores, 95)
            )
        return (scores > threshold).astype(int)

    def fit_predict(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> ModelResult:
        """Fit on training data and score test data."""
        import time

        start = time.time()
        self.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        scores = self.score(X_test)
        inference_time = time.time() - start

        return ModelResult(
            scores=scores,
            train_time=train_time,
            inference_time=inference_time,
            metadata={"model_name": self.name, "seed": self.seed},
        )

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed}

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v}" for k, v in self.get_params().items()
        )
        return f"{self.__class__.__name__}({params})"


class StaticDetector(BaseAnomalyDetector):
    """Base for detectors operating on fixed-size UBFS vectors."""

    def validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X


class TemporalDetector(BaseAnomalyDetector):
    """Base for detectors operating on sequences of UBFS vectors."""

    def validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        elif X.ndim != 3:
            raise ValueError(
                f"Expected 3D input, got shape {X.shape}"
            )
        return X
