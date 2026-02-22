"""Extract UBFS features from CMU-CERT dataset.

Maps the daily behavioural features from the MSc thesis pipeline
into the Unified Behavioural Feature Schema. This extractor
operates on the output of the existing feature engineering
code (daily_features DataFrame).

Feature Mapping:
    CMU-CERT daily features (~21 columns per user-day)
    → UBFS vector (20 dimensions)

Some UBFS slots require derived features not directly present
in the CMU-CERT output (e.g., peer_distance, sequence_entropy).
These are computed here from the daily features DataFrame.
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from .ubfs_schema import (
    FEATURE_DEFINITIONS,
    FeatureCategory,
    UBFSConfig,
    UBFSVector,
    ubfs_feature_names,
)


class CERTFeatureExtractor:
    """Extracts UBFS vectors from CMU-CERT daily features.

    Expects input from the MSc feature engineering pipeline:
    a Polars DataFrame with columns like logon_count,
    after_hours_logons, unique_pcs, etc.
    """

    def __init__(self):
        self.config = UBFSConfig()
        self._peer_baselines: Optional[Dict[str, np.ndarray]] = None

    def extract_user_day(
        self, row: Dict, peer_baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract a single UBFS vector from one user-day row.

        Args:
            row: Dictionary of daily feature values for one user-day.
            peer_baseline: Mean UBFS vector for the user's peer group
                (used for deviation features). None = use zeros.

        Returns:
            UBFS vector as numpy array of shape (total_dim,).
        """
        features = np.zeros(self.config.total_dim, dtype=np.float32)
        slices = self.config.category_slices

        # TEMPORAL
        s = slices[FeatureCategory.TEMPORAL]
        features[s.start + 0] = row.get("first_logon_hour", 0.0)
        last_logoff = row.get("last_logoff_hour", 0.0)
        first_logon = row.get("first_logon_hour", 0.0)
        features[s.start + 1] = max(last_logoff - first_logon, 0.0)
        logon_count = row.get("logon_count", 1)
        ah_logons = row.get("after_hours_logons", 0)
        features[s.start + 2] = (
            ah_logons / max(logon_count, 1)
        )
        features[s.start + 3] = float(
            row.get("day_of_week", 0) >= 5
        )

        # FREQUENCY
        s = slices[FeatureCategory.FREQUENCY]
        features[s.start + 0] = row.get("logon_count", 0.0)
        features[s.start + 1] = row.get("emails_sent", 0.0)
        features[s.start + 2] = row.get("device_activity", 0.0)
        # Event rate z-score filled during batch normalisation
        features[s.start + 3] = 0.0

        # VOLUME
        s = slices[FeatureCategory.VOLUME]
        features[s.start + 0] = row.get("attachment_size", 0.0)
        features[s.start + 1] = row.get("total_recipients", 0.0)
        features[s.start + 2] = 0.0  # Std computed at batch level

        # SCOPE
        s = slices[FeatureCategory.SCOPE]
        features[s.start + 0] = row.get("unique_pcs", 0.0)
        features[s.start + 1] = row.get("unique_domains", 0.0)
        pc_ratio = (
            row.get("unique_pcs", 0) / max(logon_count, 1)
        )
        features[s.start + 2] = pc_ratio

        # SEQUENCE (requires temporal context — placeholder)
        s = slices[FeatureCategory.SEQUENCE]
        features[s.start + 0] = 0.0  # Filled by extract_batch
        features[s.start + 1] = 0.0
        features[s.start + 2] = 0.0

        # DEVIATION
        s = slices[FeatureCategory.DEVIATION]
        if peer_baseline is not None:
            features[s.start + 0] = float(
                np.linalg.norm(features[:s.start] - peer_baseline[:s.start])
            )
        features[s.start + 1] = 0.0  # Filled by extract_batch

        # PRIVILEGE
        s = slices[FeatureCategory.PRIVILEGE]
        features[s.start + 0] = 0.0  # CMU-CERT lacks explicit roles

        return features

    def extract_batch(
        self, df: pl.DataFrame
    ) -> tuple[np.ndarray, List[str], List[str]]:
        """Extract UBFS vectors for all user-days in a DataFrame.

        Vectorized implementation: maps DataFrame columns directly
        to UBFS slots using numpy operations instead of row-by-row
        Python loops.

        Args:
            df: Daily features DataFrame with columns user_id,
                date, and feature columns.

        Returns:
            Tuple of:
                X: UBFS matrix (n_samples, ubfs_dim)
                entity_ids: List of user_ids
                timestamps: List of date strings
        """
        # Normalize column names: CMU-CERT uses "user"/"day",
        # this code expects "user_id"/"date"
        rename_map = {}
        if "user" in df.columns and "user_id" not in df.columns:
            rename_map["user"] = "user_id"
        if "day" in df.columns and "date" not in df.columns:
            rename_map["day"] = "date"
        if rename_map:
            df = df.rename(rename_map)

        n = len(df)
        X = np.zeros((n, self.config.total_dim), dtype=np.float32)
        slices = self.config.category_slices

        def _col(name: str) -> np.ndarray:
            """Get column as numpy, zeros if missing."""
            if name in df.columns:
                arr = df[name].to_numpy()
                return np.nan_to_num(arr, nan=0.0).astype(np.float32)
            return np.zeros(n, dtype=np.float32)

        # --- Per-row features (vectorized) ---

        # TEMPORAL
        s = slices[FeatureCategory.TEMPORAL]
        first_logon = _col("first_logon_hour")
        last_logoff = _col("last_logoff_hour")
        logon_count = np.maximum(_col("logon_count"), 1.0)
        ah_logons = _col("after_hours_logons")
        X[:, s.start + 0] = first_logon
        X[:, s.start + 1] = np.maximum(last_logoff - first_logon, 0.0)
        X[:, s.start + 2] = ah_logons / logon_count
        dow = _col("day_of_week")
        X[:, s.start + 3] = (dow >= 5).astype(np.float32)

        # FREQUENCY
        s = slices[FeatureCategory.FREQUENCY]
        X[:, s.start + 0] = _col("logon_count")
        X[:, s.start + 1] = _col("emails_sent")
        X[:, s.start + 2] = _col("device_activity")

        # VOLUME
        s = slices[FeatureCategory.VOLUME]
        X[:, s.start + 0] = _col("attachment_size")
        X[:, s.start + 1] = _col("total_recipients")

        # SCOPE
        s = slices[FeatureCategory.SCOPE]
        unique_pcs = _col("unique_pcs")
        X[:, s.start + 0] = unique_pcs
        X[:, s.start + 1] = _col("unique_domains")
        X[:, s.start + 2] = unique_pcs / logon_count

        # --- Batch-level features (vectorized) ---

        # Event rate z-score (FREQUENCY slot 3)
        freq_s = slices[FeatureCategory.FREQUENCY]
        total_events = X[:, freq_s.start] + X[:, freq_s.start + 1]
        std_rate = np.std(total_events)
        if std_rate > 0:
            X[:, freq_s.start + 3] = (
                (total_events - np.mean(total_events)) / std_rate
            )

        # Volume variability (VOLUME slot 2): per-user std
        vol_s = slices[FeatureCategory.VOLUME]
        if "attachment_size" in df.columns:
            user_stds = (
                df.group_by("user_id")
                .agg(pl.col("attachment_size").std().alias("vol_std"))
            )
            vol_df = df.select("user_id").join(
                user_stds, on="user_id", how="left"
            )
            X[:, vol_s.start + 2] = np.nan_to_num(
                vol_df["vol_std"].to_numpy(), nan=0.0
            ).astype(np.float32)

        # Sequence entropy (SEQUENCE slot 0): vectorized
        seq_s = slices[FeatureCategory.SEQUENCE]
        counts = np.stack([
            X[:, freq_s.start + j] for j in range(3)
        ], axis=1)  # (n, 3)
        totals = counts.sum(axis=1, keepdims=True)
        totals = np.maximum(totals, 1e-10)
        probs = counts / totals
        # Entropy: -sum(p * log2(p)) for p > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -np.sum(probs * log_probs, axis=1)
        X[:, seq_s.start + 0] = entropy.astype(np.float32)

        # Peer distance (DEVIATION slot 0): vectorized
        dev_s = slices[FeatureCategory.DEVIATION]
        prefix = X[:, :dev_s.start]
        global_mean = np.mean(prefix, axis=0, keepdims=True)
        X[:, dev_s.start + 0] = np.linalg.norm(
            prefix - global_mean, axis=1
        ).astype(np.float32)

        # Self-deviation (DEVIATION slot 1): per-user mean distance
        if "user_id" in df.columns:
            uid_arr = df["user_id"].to_numpy()
            unique_uids, inverse = np.unique(
                uid_arr, return_inverse=True
            )
            # Compute per-user mean of prefix features
            user_sums = np.zeros(
                (len(unique_uids), dev_s.start), dtype=np.float64
            )
            user_counts = np.zeros(len(unique_uids), dtype=np.int64)
            np.add.at(user_sums, inverse, prefix.astype(np.float64))
            np.add.at(user_counts, inverse, 1)
            user_means = user_sums / np.maximum(
                user_counts[:, None], 1
            )
            # Distance from each row to its user's mean
            row_user_means = user_means[inverse]
            X[:, dev_s.start + 1] = np.linalg.norm(
                prefix - row_user_means, axis=1
            ).astype(np.float32)

        # Entity IDs and timestamps
        entity_ids = (
            df["user_id"].cast(pl.Utf8).to_list()
            if "user_id" in df.columns
            else [f"user_{i}" for i in range(n)]
        )
        timestamps = (
            df["date"].cast(pl.Utf8).to_list()
            if "date" in df.columns
            else [""] * n
        )

        return X, entity_ids, timestamps

    def to_ubfs_vectors(
        self,
        X: np.ndarray,
        entity_ids: List[str],
        timestamps: List[str],
    ) -> List[UBFSVector]:
        """Wrap raw arrays into UBFSVector objects."""
        return [
            UBFSVector(
                values=X[i],
                entity_id=entity_ids[i],
                domain="cert",
                timestamp=timestamps[i],
            )
            for i in range(len(X))
        ]
