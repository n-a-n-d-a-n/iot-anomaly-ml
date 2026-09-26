"""Tests for Isolation Forest anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.isolation_forest import IsolationForestDetector


@pytest.fixture()
def sample_data() -> tuple[np.ndarray, list[str]]:
    """Create synthetic training data with clear anomalies."""
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (200, 3))
    anomalies = rng.normal(5, 0.5, (10, 3))
    X = np.vstack([normal, anomalies])
    feature_names = ["sensor_a", "sensor_b", "sensor_c"]
    return X, feature_names


@pytest.fixture()
def fitted_detector(sample_data: tuple[np.ndarray, list[str]]) -> IsolationForestDetector:
    X, names = sample_data
    detector = IsolationForestDetector(contamination=0.05, n_estimators=50, random_state=42)
    detector.fit(X, names)
    return detector


class TestFit:
    def test_creates_multivariate_model(self, fitted_detector: IsolationForestDetector) -> None:
        assert fitted_detector.multivariate_model is not None

    def test_creates_univariate_models(self, fitted_detector: IsolationForestDetector) -> None:
        assert len(fitted_detector.univariate_models) == 3
        for name in ["sensor_a", "sensor_b", "sensor_c"]:
            assert name in fitted_detector.univariate_models

    def test_stores_feature_names(self, fitted_detector: IsolationForestDetector) -> None:
        assert fitted_detector.feature_names == ["sensor_a", "sensor_b", "sensor_c"]


class TestPredict:
    def test_output_shape(
        self,
        fitted_detector: IsolationForestDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        preds = fitted_detector.predict(X)
        assert preds.shape == (len(X),)

    def test_binary_output(
        self,
        fitted_detector: IsolationForestDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        preds = fitted_detector.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_detects_obvious_anomalies(
        self,
        fitted_detector: IsolationForestDetector,
    ) -> None:
        # Points far from training distribution should be flagged
        far_points = np.full((5, 3), 10.0)
        preds = fitted_detector.predict(far_points)
        assert preds.sum() > 0


class TestScoreSamples:
    def test_scores_in_unit_range(
        self,
        fitted_detector: IsolationForestDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        scores = fitted_detector.score_samples(X)
        assert scores.min() >= -0.01
        assert scores.max() <= 1.01

    def test_anomalies_score_higher(self, fitted_detector: IsolationForestDetector) -> None:
        normal = np.zeros((50, 3))
        anomalous = np.full((50, 3), 8.0)
        combined = np.vstack([normal, anomalous])
        scores = fitted_detector.score_samples(combined)
        normal_scores = scores[:50]
        anomalous_scores = scores[50:]
        assert anomalous_scores.mean() > normal_scores.mean()


class TestFeatureContributions:
    def test_returns_all_features(
        self,
        fitted_detector: IsolationForestDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        contribs = fitted_detector.get_feature_contributions(X)
        assert set(contribs.keys()) == {"sensor_a", "sensor_b", "sensor_c"}

    def test_contribution_scores_in_range(
        self,
        fitted_detector: IsolationForestDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        contribs = fitted_detector.get_feature_contributions(X)
        for scores in contribs.values():
            assert scores.min() >= -0.01
            assert scores.max() <= 1.01


class TestSaveLoad:
    def test_roundtrip(
        self,
        fitted_detector: IsolationForestDetector,
        sample_data: tuple[np.ndarray, list[str]],
        tmp_path,
    ) -> None:
        X, _ = sample_data
        path = str(tmp_path / "model.joblib")
        fitted_detector.save(path)
        loaded = IsolationForestDetector.load(path)

        np.testing.assert_array_equal(
            fitted_detector.predict(X),
            loaded.predict(X),
        )
        np.testing.assert_array_almost_equal(
            fitted_detector.score_samples(X),
            loaded.score_samples(X),
        )

    def test_preserves_config(
        self,
        fitted_detector: IsolationForestDetector,
        tmp_path,
    ) -> None:
        path = str(tmp_path / "model.joblib")
        fitted_detector.save(path)
        loaded = IsolationForestDetector.load(path)
        assert loaded.contamination == fitted_detector.contamination
        assert loaded.n_estimators == fitted_detector.n_estimators
        assert loaded.feature_names == fitted_detector.feature_names
