"""Tests for UBFS schema and feature extractors."""

import numpy as np
import pytest

from src.features.ubfs_schema import (
    FEATURE_DEFINITIONS,
    FeatureCategory,
    UBFSConfig,
    UBFSNormalizer,
    UBFSVector,
    ubfs_agent_mapping,
    ubfs_cert_mapping,
    ubfs_feature_names,
)


class TestUBFSConfig:
    def test_total_dim(self):
        config = UBFSConfig()
        assert config.total_dim == 20

    def test_category_dims_sum(self):
        config = UBFSConfig()
        total = sum(config.category_dims.values())
        assert total == config.total_dim

    def test_category_slices_cover_all(self):
        config = UBFSConfig()
        slices = config.category_slices
        covered = set()
        for s in slices.values():
            covered.update(range(s.start, s.stop))
        assert covered == set(range(config.total_dim))

    def test_all_categories_present(self):
        config = UBFSConfig()
        for cat in FeatureCategory:
            assert cat in config.category_dims


class TestUBFSVector:
    def test_zeros_shape(self):
        v = UBFSVector.zeros("test_user", "cert")
        assert v.values.shape == (20,)
        assert np.all(v.values == 0.0)

    def test_domain_validation(self):
        with pytest.raises(ValueError, match="Domain"):
            UBFSVector(
                values=np.zeros(20, dtype=np.float32),
                entity_id="test",
                domain="invalid",
            )

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="dimensions"):
            UBFSVector(
                values=np.zeros(10, dtype=np.float32),
                entity_id="test",
                domain="cert",
            )

    def test_round_trip_serialisation(self):
        v = UBFSVector(
            values=np.random.randn(20).astype(np.float32),
            entity_id="user123",
            domain="agent",
            timestamp="2025-01-15",
        )
        d = v.to_dict()
        v2 = UBFSVector.from_dict(d)
        np.testing.assert_allclose(v.values, v2.values, atol=1e-6)
        assert v2.entity_id == "user123"
        assert v2.domain == "agent"

    def test_get_category(self):
        config = UBFSConfig()
        v = UBFSVector.zeros("test", "cert")
        temporal = v.get_category(FeatureCategory.TEMPORAL)
        assert len(temporal) == config.category_dims[
            FeatureCategory.TEMPORAL
        ]


class TestUBFSNormalizer:
    def test_zscore_normalisation(self):
        X = np.random.randn(100, 20).astype(np.float32) * 5 + 3
        norm = UBFSNormalizer(method="zscore")
        X_norm = norm.fit_transform(X)
        np.testing.assert_allclose(
            np.mean(X_norm, axis=0), 0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            np.std(X_norm, axis=0), 1.0, atol=1e-5
        )

    def test_fit_then_transform(self):
        X_train = np.random.randn(100, 20).astype(np.float32)
        X_test = np.random.randn(50, 20).astype(np.float32)

        norm = UBFSNormalizer()
        norm.fit(X_train)
        X_test_norm = norm.transform(X_test)
        assert X_test_norm.shape == (50, 20)

    def test_transform_before_fit_raises(self):
        norm = UBFSNormalizer()
        with pytest.raises(RuntimeError, match="fitted"):
            norm.transform(np.zeros((10, 20)))


class TestFeatureMapping:
    def test_cert_mapping_complete(self):
        mapping = ubfs_cert_mapping()
        names = ubfs_feature_names()
        assert set(mapping.keys()) == set(names)

    def test_agent_mapping_complete(self):
        mapping = ubfs_agent_mapping()
        names = ubfs_feature_names()
        assert set(mapping.keys()) == set(names)

    def test_feature_names_length(self):
        names = ubfs_feature_names()
        config = UBFSConfig()
        assert len(names) == config.total_dim

    def test_no_duplicate_names(self):
        names = ubfs_feature_names()
        assert len(names) == len(set(names))


class TestCERTExtractor:
    """Test CERTFeatureExtractor handles CMU-CERT column names."""

    def test_extract_batch_with_user_column(self):
        """CMU-CERT uses 'user'/'day' not 'user_id'/'date'."""
        import polars as pl
        from src.features.cert_extractor import CERTFeatureExtractor

        df = pl.DataFrame({
            "user": ["u1", "u1", "u2"],
            "day": ["2010-01-01", "2010-01-02", "2010-01-01"],
            "logon_count": [5, 3, 7],
            "first_logon_hour": [8.0, 9.0, 7.0],
            "last_logoff_hour": [17.0, 18.0, 16.0],
            "after_hours_logons": [0, 1, 0],
            "emails_sent": [10, 5, 8],
            "device_activity": [2, 1, 3],
            "attachment_size": [100.0, 200.0, 50.0],
            "total_recipients": [3, 2, 5],
            "unique_pcs": [1, 2, 1],
            "unique_domains": [5, 3, 8],
        })

        ext = CERTFeatureExtractor()
        X, entity_ids, timestamps = ext.extract_batch(df)

        assert X.shape == (3, ext.config.total_dim)
        assert entity_ids == ["u1", "u1", "u2"]
        assert timestamps == ["2010-01-01", "2010-01-02", "2010-01-01"]

    def test_extract_batch_with_user_id_column(self):
        """Also works with 'user_id'/'date' naming convention."""
        import polars as pl
        from src.features.cert_extractor import CERTFeatureExtractor

        df = pl.DataFrame({
            "user_id": ["u1", "u2"],
            "date": ["2010-01-01", "2010-01-01"],
            "logon_count": [5, 7],
            "first_logon_hour": [8.0, 7.0],
            "last_logoff_hour": [17.0, 16.0],
        })

        ext = CERTFeatureExtractor()
        X, entity_ids, timestamps = ext.extract_batch(df)

        assert X.shape == (2, ext.config.total_dim)
        assert entity_ids == ["u1", "u2"]
        assert timestamps == ["2010-01-01", "2010-01-01"]
