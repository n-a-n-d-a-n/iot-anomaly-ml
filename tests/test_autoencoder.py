"""Tests for PyTorch autoencoder anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.autoencoder import Autoencoder, AutoencoderDetector


@pytest.fixture()
def sample_data() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 4))
    return X, ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def fitted_detector(sample_data: tuple[np.ndarray, list[str]]) -> AutoencoderDetector:
    X, names = sample_data
    detector = AutoencoderDetector(hidden_dims=[16, 8], epochs=5, lr=0.01, device="cpu")
    detector.fit(X, names)
    return detector


class TestAutoencoderModule:
    def test_forward_preserves_shape(self) -> None:
        import torch

        model = Autoencoder(input_dim=4, hidden_dims=[8, 4])
        model.eval()
        x = torch.randn(10, 4)
        out = model(x)
        assert out.shape == (10, 4)

    def test_encode_reduces_dim(self) -> None:
        import torch

        model = Autoencoder(input_dim=4, hidden_dims=[8, 2])
        model.eval()
        x = torch.randn(10, 4)
        encoded = model.encode(x)
        assert encoded.shape == (10, 2)


class TestFit:
    def test_model_created(self, fitted_detector: AutoencoderDetector) -> None:
        assert fitted_detector.model is not None

    def test_threshold_set(self, fitted_detector: AutoencoderDetector) -> None:
        assert fitted_detector.threshold > 0

    def test_feature_names_stored(self, fitted_detector: AutoencoderDetector) -> None:
        assert fitted_detector.feature_names == ["f0", "f1", "f2", "f3"]


class TestPredict:
    def test_output_shape(
        self,
        fitted_detector: AutoencoderDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        preds = fitted_detector.predict(X)
        assert preds.shape == (len(X),)

    def test_binary_output(
        self,
        fitted_detector: AutoencoderDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        preds = fitted_detector.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})


class TestScoreSamples:
    def test_scores_in_unit_range(
        self,
        fitted_detector: AutoencoderDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        scores = fitted_detector.score_samples(X)
        assert scores.min() >= -0.01
        assert scores.max() <= 1.01

    def test_output_shape(
        self,
        fitted_detector: AutoencoderDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        scores = fitted_detector.score_samples(X)
        assert scores.shape == (len(X),)


class TestFeatureContributions:
    def test_returns_all_features(
        self,
        fitted_detector: AutoencoderDetector,
        sample_data: tuple[np.ndarray, list[str]],
    ) -> None:
        X, _ = sample_data
        contribs = fitted_detector.get_feature_contributions(X)
        assert set(contribs.keys()) == {"f0", "f1", "f2", "f3"}

    def test_contribution_values_in_range(
        self,
        fitted_detector: AutoencoderDetector,
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
        fitted_detector: AutoencoderDetector,
        sample_data: tuple[np.ndarray, list[str]],
        tmp_path,
    ) -> None:
        X, _ = sample_data
        path = str(tmp_path / "ae.pt")
        fitted_detector.save(path)
        loaded = AutoencoderDetector.load(path, device="cpu")

        np.testing.assert_array_almost_equal(
            fitted_detector.score_samples(X),
            loaded.score_samples(X),
            decimal=5,
        )

    def test_preserves_config(
        self,
        fitted_detector: AutoencoderDetector,
        tmp_path,
    ) -> None:
        path = str(tmp_path / "ae.pt")
        fitted_detector.save(path)
        loaded = AutoencoderDetector.load(path, device="cpu")
        assert loaded.hidden_dims == fitted_detector.hidden_dims
        assert loaded.feature_names == fitted_detector.feature_names
        assert abs(loaded.threshold - fitted_detector.threshold) < 1e-6
