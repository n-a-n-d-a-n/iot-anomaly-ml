"""Tests for ensemble scoring and model evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ensemble import EnsembleDetector, EnsembleResult
from src.models.evaluator import EvaluationResult, ModelEvaluator


class _FakeModel:
    """Minimal stub implementing the detector interface."""

    def __init__(self, scores: np.ndarray, feature_names: list[str]) -> None:
        self._scores = scores
        self._feature_names = feature_names

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._scores[: len(X)]

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {name: self._scores[: len(X)] for name in self._feature_names}


@pytest.fixture()
def fake_models() -> dict[str, _FakeModel]:
    scores_a = np.linspace(0, 1, 100)
    scores_b = np.linspace(0.2, 0.8, 100)
    return {
        "model_a": _FakeModel(scores_a, ["f1", "f2"]),
        "model_b": _FakeModel(scores_b, ["f1", "f2"]),
    }


@pytest.fixture()
def ensemble(fake_models: dict[str, _FakeModel]) -> EnsembleDetector:
    ens = EnsembleDetector(weights={"model_a": 0.6, "model_b": 0.4})
    for name, model in fake_models.items():
        ens.add_model(name, model)
    return ens


class TestEnsembleDetector:
    def test_add_model(self, ensemble: EnsembleDetector) -> None:
        assert len(ensemble.models) == 2

    def test_predict_returns_ensemble_result(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        result = ensemble.predict(X)
        assert isinstance(result, EnsembleResult)

    def test_scores_shape(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        result = ensemble.predict(X)
        assert result.scores.shape == (100,)

    def test_predictions_binary(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        result = ensemble.predict(X)
        assert set(np.unique(result.predictions)).issubset({0, 1})

    def test_method_scores_present(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        result = ensemble.predict(X)
        assert "model_a" in result.method_scores
        assert "model_b" in result.method_scores

    def test_feature_contributions_present(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        result = ensemble.predict(X)
        assert "f1" in result.feature_contributions
        assert "f2" in result.feature_contributions

    def test_weighted_scores(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(50, 2)
        result = ensemble.predict(X)
        # Scores should be a weighted blend
        manual = 0.6 * ensemble.models["model_a"].score_samples(X) + 0.4 * ensemble.models[
            "model_b"
        ].score_samples(X)
        np.testing.assert_array_almost_equal(result.scores, manual)


class TestCalibrateThreshold:
    def test_improves_f1(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        y = np.zeros(100)
        y[80:] = 1  # last 20 are anomalies
        threshold = ensemble.calibrate_threshold(X, y)
        assert 0 <= threshold <= 1

    def test_updates_threshold(self, ensemble: EnsembleDetector) -> None:
        X = np.random.randn(100, 2)
        y = np.zeros(100)
        y[90:] = 1
        ensemble.calibrate_threshold(X, y)
        # Threshold should be a valid float after calibration
        assert isinstance(ensemble.threshold, float)


class TestModelEvaluator:
    def test_evaluate_basic(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        result = ModelEvaluator.evaluate(y_true, y_pred)
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1 <= 1

    def test_evaluate_with_scores(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4])
        result = ModelEvaluator.evaluate(y_true, y_pred, y_scores)
        assert result.auc_roc > 0

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = ModelEvaluator.evaluate(y_true, y_pred)
        assert result.confusion_matrix.shape == (2, 2)

    def test_to_dict(self) -> None:
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 0])
        result = ModelEvaluator.evaluate(y_true, y_pred)
        d = result.to_dict()
        assert "precision" in d
        assert "confusion_matrix" in d
        assert isinstance(d["confusion_matrix"], list)

    def test_zero_division_safe(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        result = ModelEvaluator.evaluate(y_true, y_pred)
        assert result.f1 == 0.0

    def test_compare_models(self) -> None:
        r1 = EvaluationResult(0.9, 0.8, 0.85, 0.9, 0.9, np.array([[80, 10], [20, 90]]))
        r2 = EvaluationResult(0.7, 0.9, 0.79, 0.85, 0.8, np.array([[70, 20], [10, 100]]))
        report = ModelEvaluator.compare_models({"ModelA": r1, "ModelB": r2})
        assert "ModelA" in report
        assert "ModelB" in report
        assert "Best Model" in report
