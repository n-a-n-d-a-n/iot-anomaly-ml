"""Gaussian Mixture Model anomaly detector.

Fits a GMM to normal data and scores new samples by their negative
log-likelihood under the fitted mixture — low likelihood (high score)
indicates a point that doesn't fit the learned normal distribution.
Unlike Isolation Forest / DBSCAN, this gives a smooth probabilistic
severity score rather than a purely geometric outlier notion, which is
useful for ranking anomalies by how "surprising" they are.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GMMDetector:
    """Anomaly detector using a Gaussian Mixture Model.

    Args:
        n_components: Number of Gaussian components in the mixture.
        covariance_type: Covariance parameterization ("full", "diag", "tied", "spherical").
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: GaussianMixture | None = None
        self.feature_names: list[str] = []
        # Score normalisation range and anomaly threshold, calibrated at fit time
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._anomaly_threshold: float = 0.95

    def fit(
        self, X: np.ndarray, feature_names: list[str], threshold_percentile: float = 95.0
    ) -> GMMDetector:
        """Fit a Gaussian Mixture Model on (assumed mostly normal) training data.

        Args:
            X: Training feature array of shape ``(n_samples, n_features)``.
            feature_names: Names corresponding to each feature column.
            threshold_percentile: Percentile of training scores used to set the
                anomaly cutoff (e.g. 95 flags the top 5% most surprising points).

        Returns:
            ``self`` for method chaining.
        """
        self.feature_names = feature_names

        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            reg_covar=1e-4,
        )
        self.model.fit(X)

        # Calibrate score range and anomaly threshold on the training set
        raw = -self.model.score_samples(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        score_range = (self._score_max - self._score_min) + 1e-10
        normed = (raw - self._score_min) / score_range
        self._anomaly_threshold = float(np.percentile(normed, threshold_percentile))

        logger.info(
            "Fitted GMM (%d components) on %d features, %d samples, threshold=%.4f",
            self.n_components,
            len(feature_names),
            len(X),
            self._anomaly_threshold,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies: 1 = anomaly, 0 = normal.

        A sample is flagged anomalous if its normalised score exceeds the
        threshold calibrated during :meth:`fit`.

        Args:
            X: Feature array.

        Returns:
            Binary prediction array.
        """
        norm = self.score_samples(X)
        return (norm >= self._anomaly_threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute normalised anomaly scores in [0, 1] (higher = more anomalous).

        Args:
            X: Feature array.

        Returns:
            Score array of shape ``(n_samples,)``.
        """
        raw_scores = -self.model.score_samples(X)
        score_range = (self._score_max - self._score_min) + 1e-10
        normed = (raw_scores - self._score_min) / score_range
        return np.clip(normed, 0.0, 1.0)

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Approximate per-feature contribution via leave-one-out likelihood delta.

        For explainability parity with the other detectors: for each feature,
        we perturb it to the population mean and measure how much the sample's
        anomaly score drops — a large drop means that feature was driving the flag.

        Args:
            X: Feature array.

        Returns:
            Dict mapping feature name to a normalised contribution score array.
        """
        base_scores = self.score_samples(X)
        means = X.mean(axis=0)
        contributions: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.feature_names):
            X_perturbed = X.copy()
            X_perturbed[:, i] = means[i]
            perturbed_scores = self.score_samples(X_perturbed)
            delta = np.clip(base_scores - perturbed_scores, 0.0, None)
            contributions[name] = delta
        return contributions

    def save(self, path: str) -> None:
        """Persist the model to disk.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "anomaly_threshold": self._anomaly_threshold,
                "config": {
                    "n_components": self.n_components,
                    "covariance_type": self.covariance_type,
                },
            },
            path,
        )
        logger.info("Saved GMM model to %s", path)

    @classmethod
    def load(cls, path: str) -> GMMDetector:
        """Load a persisted model from disk.

        Args:
            path: File path saved by :meth:`save`.

        Returns:
            Restored detector instance.
        """
        data = joblib.load(path)
        detector = cls(
            n_components=data["config"]["n_components"],
            covariance_type=data["config"]["covariance_type"],
        )
        detector.model = data["model"]
        detector.feature_names = data["feature_names"]
        detector._score_min = data["score_min"]
        detector._score_max = data["score_max"]
        detector._anomaly_threshold = data.get("anomaly_threshold", 0.95)
        return detector
