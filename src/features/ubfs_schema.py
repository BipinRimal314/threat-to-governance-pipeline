"""Unified Behavioural Feature Schema (UBFS).

The UBFS defines a common feature space that bridges insider threat
detection (CMU-CERT employee activity logs) and AI agent monitoring
(OpenTelemetry agent traces). Both domain-specific extractors map
their raw data into this shared representation, enabling the same
anomaly detection models to operate on either domain.

Feature Categories:
    TEMPORAL   - Timing patterns (when activity occurs)
    FREQUENCY  - Event rates (how often things happen)
    VOLUME     - Data quantities (how much data moves)
    SCOPE      - Breadth of access (how widely entity operates)
    SEQUENCE   - Pattern complexity (how predictable behaviour is)
    DEVIATION  - Departure from baseline (how different from peers)
    PRIVILEGE  - Permission usage patterns (what access level used)

Each category has a fixed number of feature slots. Domain-specific
extractors fill these slots from their respective data sources.
Features that have no natural mapping in a domain are set to 0.0
(the normalised neutral value under z-score normalisation).

Design Rationale:
    The structural analogy between insider threats and rogue agents
    is encoded in this schema: both involve an entity with legitimate
    access deviating from expected behavioural patterns. The UBFS
    makes this analogy explicit and testable.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class FeatureCategory(Enum):
    """Behavioural feature categories shared across domains."""

    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    VOLUME = "volume"
    SCOPE = "scope"
    SEQUENCE = "sequence"
    DEVIATION = "deviation"
    PRIVILEGE = "privilege"


# Feature definitions per category.
# Each tuple: (unified_name, cert_source, agent_source, description)
# cert_source/agent_source are the raw field names from each domain.
# None means the feature has no direct mapping in that domain.

FEATURE_DEFINITIONS: Dict[FeatureCategory, List[Tuple[str, Optional[str], Optional[str], str]]] = {
    FeatureCategory.TEMPORAL: [
        (
            "activity_hour_mean",
            "first_logon_hour",
            "trace_start_hour",
            "Mean hour of activity onset",
        ),
        (
            "session_duration_norm",
            "session_duration",
            "trace_duration",
            "Normalised session/trace duration",
        ),
        (
            "after_hours_ratio",
            "after_hours_logons",
            "off_schedule_ratio",
            "Fraction of activity outside normal hours",
        ),
        (
            "weekend_activity_flag",
            "is_weekend",
            "is_off_schedule",
            "Activity during non-standard periods",
        ),
    ],
    FeatureCategory.FREQUENCY: [
        (
            "primary_event_count",
            "logon_count",
            "tool_call_count",
            "Count of primary action type",
        ),
        (
            "secondary_event_count",
            "email_count",
            "llm_call_count",
            "Count of secondary action type",
        ),
        (
            "peripheral_event_count",
            "device_activity",
            "retry_count",
            "Count of peripheral/auxiliary events",
        ),
        (
            "event_rate_zscore",
            "daily_event_rate_z",
            "span_rate_z",
            "Z-scored event rate vs. baseline",
        ),
    ],
    FeatureCategory.VOLUME: [
        (
            "data_volume_norm",
            "attachment_size",
            "total_tokens",
            "Normalised data volume transferred",
        ),
        (
            "output_count_norm",
            "total_recipients",
            "output_artifact_count",
            "Normalised count of outputs generated",
        ),
        (
            "volume_variability",
            "attachment_size_std",
            "token_volume_std",
            "Std dev of volume across sub-events",
        ),
    ],
    FeatureCategory.SCOPE: [
        (
            "resource_breadth",
            "unique_pcs",
            "unique_tools_invoked",
            "Number of distinct resources accessed",
        ),
        (
            "target_breadth",
            "unique_domains",
            "unique_endpoints",
            "Number of distinct targets contacted",
        ),
        (
            "breadth_ratio",
            "pc_per_logon_ratio",
            "tool_breadth_ratio",
            "Breadth normalised by event count",
        ),
    ],
    FeatureCategory.SEQUENCE: [
        (
            "action_entropy",
            "action_sequence_entropy",
            "span_sequence_entropy",
            "Shannon entropy of action sequence",
        ),
        (
            "transition_novelty",
            "transition_prob_deviation",
            "bigram_novelty",
            "Fraction of unusual state transitions",
        ),
        (
            "repetition_score",
            "action_repetition_rate",
            "tool_repetition_rate",
            "Degree of repetitive behaviour",
        ),
    ],
    FeatureCategory.DEVIATION: [
        (
            "peer_distance",
            "peer_group_distance",
            "agent_type_distance",
            "Distance from peer/type baseline",
        ),
        (
            "self_deviation",
            "self_baseline_deviation",
            "trace_baseline_deviation",
            "Deviation from own historical baseline",
        ),
    ],
    FeatureCategory.PRIVILEGE: [
        (
            "privilege_deviation_index",
            "access_level_deviation",
            "permission_scope_deviation",
            "Deviation of privilege level from norm",
        ),
    ],
}


@dataclass(frozen=True)
class UBFSConfig:
    """Configuration for the Unified Behavioural Feature Schema."""

    normalization: str = "zscore"

    @property
    def total_dim(self) -> int:
        """Total number of features in a UBFS vector."""
        return sum(
            len(features)
            for features in FEATURE_DEFINITIONS.values()
        )

    @property
    def category_dims(self) -> Dict[FeatureCategory, int]:
        """Number of features per category."""
        return {
            cat: len(feats)
            for cat, feats in FEATURE_DEFINITIONS.items()
        }

    @property
    def category_slices(self) -> Dict[FeatureCategory, slice]:
        """Index slices for each category in the UBFS vector."""
        slices = {}
        offset = 0
        for cat in FeatureCategory:
            n = len(FEATURE_DEFINITIONS[cat])
            slices[cat] = slice(offset, offset + n)
            offset += n
        return slices


def ubfs_feature_names() -> List[str]:
    """Return ordered list of all UBFS feature names."""
    names = []
    for cat in FeatureCategory:
        for name, _, _, _ in FEATURE_DEFINITIONS[cat]:
            names.append(name)
    return names


def ubfs_feature_descriptions() -> Dict[str, str]:
    """Return mapping of feature names to descriptions."""
    descs = {}
    for cat in FeatureCategory:
        for name, _, _, desc in FEATURE_DEFINITIONS[cat]:
            descs[name] = desc
    return descs


def ubfs_cert_mapping() -> Dict[str, Optional[str]]:
    """Return mapping of UBFS names to CMU-CERT source fields."""
    mapping = {}
    for cat in FeatureCategory:
        for unified, cert, _, _ in FEATURE_DEFINITIONS[cat]:
            mapping[unified] = cert
    return mapping


def ubfs_agent_mapping() -> Dict[str, Optional[str]]:
    """Return mapping of UBFS names to agent trace source fields."""
    mapping = {}
    for cat in FeatureCategory:
        for unified, _, agent, _ in FEATURE_DEFINITIONS[cat]:
            mapping[unified] = agent
    return mapping


@dataclass
class UBFSVector:
    """A single point in the Unified Behavioural Feature Schema space.

    Attributes:
        values: Feature values as a numpy array of shape (total_dim,).
        entity_id: Identifier for the entity (user_id or agent_id).
        domain: Source domain ('cert' or 'agent').
        timestamp: Time window this vector represents.
        metadata: Additional domain-specific context.
    """

    values: np.ndarray
    entity_id: str
    domain: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = field(default_factory=dict)

    _config: UBFSConfig = field(
        default_factory=UBFSConfig, repr=False
    )

    def __post_init__(self):
        if self.values.shape != (self._config.total_dim,):
            raise ValueError(
                f"UBFS vector must have {self._config.total_dim} "
                f"dimensions, got {self.values.shape}"
            )
        if self.domain not in ("cert", "agent"):
            raise ValueError(
                f"Domain must be 'cert' or 'agent', got '{self.domain}'"
            )

    def get_category(self, category: FeatureCategory) -> np.ndarray:
        """Extract feature values for a specific category."""
        s = self._config.category_slices[category]
        return self.values[s]

    def to_dict(self) -> Dict:
        """Convert to dictionary with named features."""
        names = ubfs_feature_names()
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "features": {
                name: float(val)
                for name, val in zip(names, self.values)
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "UBFSVector":
        """Construct from dictionary."""
        names = ubfs_feature_names()
        values = np.array(
            [d["features"].get(name, 0.0) for name in names],
            dtype=np.float32,
        )
        return cls(
            values=values,
            entity_id=d["entity_id"],
            domain=d["domain"],
            timestamp=d.get("timestamp"),
        )

    @classmethod
    def zeros(
        cls, entity_id: str, domain: str, **kwargs
    ) -> "UBFSVector":
        """Create a zero-valued UBFS vector (neutral baseline)."""
        config = UBFSConfig()
        return cls(
            values=np.zeros(config.total_dim, dtype=np.float32),
            entity_id=entity_id,
            domain=domain,
            **kwargs,
        )


class UBFSNormalizer:
    """Fit-transform normalizer for UBFS vectors.

    Fits statistics on training data and applies the same
    transform to test data. This prevents data leakage.
    """

    def __init__(self, method: str = "zscore"):
        self.method = method
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "UBFSNormalizer":
        """Compute normalization statistics from training data.

        Args:
            X: Training data of shape (n_samples, ubfs_dim).
        """
        if self.method == "zscore":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1.0
        elif self.method == "minmax":
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            range_ = self.max_ - self.min_
            range_[range_ == 0] = 1.0
            self.max_ = self.min_ + range_
        elif self.method == "robust":
            self.median_ = np.median(X, axis=0)
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            self.iqr_ = q3 - q1
            self.iqr_[self.iqr_ == 0] = 1.0
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalisation to data.

        Args:
            X: Data of shape (n_samples, ubfs_dim).

        Returns:
            Normalised array of same shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted first")

        if self.method == "zscore":
            return (X - self.mean_) / self.std_
        elif self.method == "minmax":
            return (X - self.min_) / (self.max_ - self.min_)
        elif self.method == "robust":
            return (X - self.median_) / self.iqr_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


if __name__ == "__main__":
    config = UBFSConfig()
    print(f"UBFS total dimensions: {config.total_dim}")
    print(f"Category dimensions: {config.category_dims}")
    print(f"Feature names: {ubfs_feature_names()}")

    # Create a sample vector
    v = UBFSVector.zeros("user001", "cert")
    print(f"\nZero vector shape: {v.values.shape}")
    print(f"Temporal slice: {v.get_category(FeatureCategory.TEMPORAL)}")

    # Test round-trip serialisation
    d = v.to_dict()
    v2 = UBFSVector.from_dict(d)
    assert np.allclose(v.values, v2.values)
    print("Round-trip serialisation: OK")
