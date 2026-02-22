"""Ensemble detector combining all three anomaly models.

Supports weighted voting, majority voting, and cascade strategies
for combining Isolation Forest, LSTM Autoencoder, and Deep
Clustering predictions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseAnomalyDetector, ModelResult


class EnsembleDetector:
    """Combines multiple anomaly detectors into a single scorer.

    Methods:
        weighted: Weighted average of normalised scores.
        majority: Binary vote from each model's threshold.
        cascade:  Sequential filtering with increasing precision.
    """

    def __init__(
        self,
        models: Dict[str, BaseAnomalyDetector],
        method: str = "weighted",
        weights: Optional[Dict[str, float]] = None,
        final_threshold: float = 0.7,
    ):
        self.models = models
        self.method = method
        self.weights = weights or {
            name: 1.0 / len(models) for name in models
        }
        self.final_threshold = final_threshold
        self.is_fitted = False

    def fit(
        self,
        X_static: Optional[np.ndarray] = None,
        X_temporal: Optional[np.ndarray] = None,
    ) -> "EnsembleDetector":
        """Fit all sub-models.

        Args:
            X_static: Training data for static detectors
                (n_samples, n_features).
            X_temporal: Training data for temporal detectors
                (n_samples, seq_len, n_features).
        """
        from .base import StaticDetector, TemporalDetector

        for name, model in self.models.items():
            if isinstance(model, TemporalDetector):
                if X_temporal is None:
                    raise ValueError(
                        f"Temporal data required for {name}"
                    )
                model.fit(X_temporal)
            elif isinstance(model, StaticDetector):
                if X_static is None:
                    raise ValueError(
                        f"Static data required for {name}"
                    )
                model.fit(X_static)
            else:
                if X_static is not None:
                    model.fit(X_static)

        self.is_fitted = True
        return self

    def score(
        self,
        X_static: Optional[np.ndarray] = None,
        X_temporal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute ensemble anomaly scores."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted first")

        scores = self._collect_scores(X_static, X_temporal)

        if self.method == "weighted":
            return self._weighted_score(scores)
        elif self.method == "majority":
            return self._majority_score(scores)
        elif self.method == "cascade":
            return self._cascade_score(scores)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(
        self,
        X_static: Optional[np.ndarray] = None,
        X_temporal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Binary ensemble predictions."""
        scores = self.score(X_static, X_temporal)
        return (scores > self.final_threshold).astype(int)

    def _collect_scores(
        self,
        X_static: Optional[np.ndarray],
        X_temporal: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Get normalised scores from each sub-model."""
        from .base import StaticDetector, TemporalDetector

        raw_scores = {}
        for name, model in self.models.items():
            if isinstance(model, TemporalDetector):
                raw_scores[name] = model.score(X_temporal)
            else:
                raw_scores[name] = model.score(X_static)

        # Min-max normalise each model's scores to [0, 1]
        normalised = {}
        for name, s in raw_scores.items():
            s_min, s_max = s.min(), s.max()
            rng = max(s_max - s_min, 1e-8)
            normalised[name] = (s - s_min) / rng

        return normalised

    def _weighted_score(
        self, scores: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Weighted average of normalised scores."""
        result = np.zeros_like(next(iter(scores.values())))
        total_weight = 0.0
        for name, s in scores.items():
            w = self.weights.get(name, 0.0)
            result += w * s
            total_weight += w
        return result / max(total_weight, 1e-8)

    def _majority_score(
        self, scores: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Fraction of models voting anomalous (above 0.5)."""
        votes = np.stack(
            [(s > 0.5).astype(float) for s in scores.values()]
        )
        return np.mean(votes, axis=0)

    def _cascade_score(
        self, scores: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Cascade: only pass samples flagged by fast model to slow.

        Uses Isolation Forest as first filter, then refines with
        LSTM and Deep Clustering.
        """
        # Start with IF scores if available, else first model
        names = list(scores.keys())
        result = scores[names[0]].copy()

        for name in names[1:]:
            # Only refine samples that passed previous threshold
            mask = result > 0.3
            if mask.any():
                result[mask] = (result[mask] + scores[name][mask]) / 2
        return result

    def get_individual_scores(
        self,
        X_static: Optional[np.ndarray] = None,
        X_temporal: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Get raw (un-normalised) scores from each model."""
        from .base import StaticDetector, TemporalDetector

        raw = {}
        for name, model in self.models.items():
            if isinstance(model, TemporalDetector):
                raw[name] = model.score(X_temporal)
            else:
                raw[name] = model.score(X_static)
        return raw
