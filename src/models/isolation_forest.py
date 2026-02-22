"""Isolation Forest for cross-domain anomaly detection.

Operates on UBFS vectors. Since Isolation Forest is feature-space
agnostic (it only requires tabular numeric input), this is the
simplest model to transfer between domains.

Adapted from MSc thesis (Thesis_work/src/models/isolation_forest.py).
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import StaticDetector


class IsolationForestDetector(StaticDetector):
    """Isolation Forest anomaly detector on UBFS feature vectors.

    Scores are the negated sklearn decision_function so that
    higher values indicate more anomalous samples.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str = "auto",
        contamination: str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__(name="IsolationForest", seed=seed)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.model_ = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "IsolationForestDetector":
        X = self.validate_input(X)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self.is_fitted = True

        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 95))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        X = self.validate_input(X)
        return -self.model_.decision_function(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
        }
