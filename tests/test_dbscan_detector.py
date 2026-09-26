"""Tests for DBSCAN-based anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.dbscan_detector import DBSCANDetector


@pytest.fixture()
def clustered_data() -> tuple[np.ndarray, list[str]]:
    """Create two well-separated clusters with some noise."""
    rng = np.random.default_rng(42)
    cluster1 = rng.normal(0, 0.3, (100, 3))
    cluster2 = rng.normal(5, 0.3, (100, 3))
    X = np.vstack([cluster1, cluster2])
    return X, ["a", "b", "c"]


@pytest.fixture()
def fitted_detector(clustered_data: tuple[np.ndarray, list[str]]) -> DBSCANDetector:
    X, names = clustered_data
    detector = DBSCANDetector(eps=1.0, min_samples=5)
    detector.fit(X, names)
    return detector


class TestFindOptimalEps:
    def test_returns_positive_value(self, clustered_data: tuple[np.ndarray, list[str]]) -> None:
        X, _ = clustered_data
        eps = DBSCANDetector.find_optimal_eps(X, min_samples=5)
        assert eps > 0

    def test_reasonable_range(self, clustered_data: tuple[np.ndarray, list[str]]) -> None:
        X, _ = clustered_data
        eps = DBSCANDetector.find_optimal_eps(X, min_samples=5)
        assert eps < 10.0


class TestFit:
    def test_creates_model(self, fitted_detector: DBSCANDetector) -> None:
        assert fitted_detector.model is not None

    def test_stores_training_data(self, fitted_detector: DBSCANDetector) -> None:
        assert fitted_detector.training_data is not None
        assert fitted_detector.training_data.shape == (200, 3)

    def test_stores_feature_names(self, fitted_detector: DBSCANDetector) -> None:
        assert fitted_detector.feature_names == ["a", "b", "c"]

    def test_auto_tune(self, clustered_data: tuple[np.ndarray, list[str]]) -> None:
        X, names = clustered_data
        detector = DBSCANDetector(eps=0.1, min_samples=5)
        detector.fit(X, names, auto_tune=True)
        assert detector.eps != 0.1  # Should have been overwritten


class TestPredict:
    def test_output_shape(
        self,
        fitted_detector: DBSCANDetector,
        clustered_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = clustered_data
        preds = fitted_detector.predict(X)
        assert preds.shape == (len(X),)

    def test_binary_output(
        self,
        fitted_detector: DBSCANDetector,
        clustered_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = clustered_data
        preds = fitted_detector.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_far_points_anomalous(self, fitted_detector: DBSCANDetector) -> None:
        far_points = np.full((10, 3), 50.0)
        preds = fitted_detector.predict(far_points)
        assert preds.sum() == 10


class TestScoreSamples:
    def test_scores_in_unit_range(
        self,
        fitted_detector: DBSCANDetector,
        clustered_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = clustered_data
        scores = fitted_detector.score_samples(X)
        assert scores.min() >= -0.01
        assert scores.max() <= 1.01


class TestFeatureContributions:
    def test_returns_all_features(
        self,
        fitted_detector: DBSCANDetector,
        clustered_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = clustered_data
        contribs = fitted_detector.get_feature_contributions(X)
        assert set(contribs.keys()) == {"a", "b", "c"}

    def test_contribution_values_in_range(
        self,
        fitted_detector: DBSCANDetector,
        clustered_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = clustered_data
        contribs = fitted_detector.get_feature_contributions(X)
        for scores in contribs.values():
            assert scores.min() >= -0.01
            assert scores.max() <= 1.01


class TestSaveLoad:
    def test_roundtrip(
        self,
        fitted_detector: DBSCANDetector,
        clustered_data: tuple[np.ndarray, list[str]],
        tmp_path,
    ) -> None:
        X, _ = clustered_data
        path = str(tmp_path / "dbscan.joblib")
        fitted_detector.save(path)
        loaded = DBSCANDetector.load(path)
        np.testing.assert_array_equal(
            fitted_detector.predict(X),
            loaded.predict(X),
        )

    def test_preserves_config(self, fitted_detector: DBSCANDetector, tmp_path) -> None:
        path = str(tmp_path / "dbscan.joblib")
        fitted_detector.save(path)
        loaded = DBSCANDetector.load(path)
        assert loaded.eps == fitted_detector.eps
        assert loaded.min_samples == fitted_detector.min_samples
        assert loaded.feature_names == fitted_detector.feature_names
