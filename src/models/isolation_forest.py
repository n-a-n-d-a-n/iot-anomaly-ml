"""Isolation Forest anomaly detector with multivariate and per-sensor univariate models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class IsolationForestDetector:
    """Anomaly detector using scikit-learn's Isolation Forest.

    Trains both a multivariate model on all features and individual
    univariate models per feature for root-cause analysis.

    Args:
        contamination: Expected fraction of anomalies in the training data.
        n_estimators: Number of trees in each Isolation Forest.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.multivariate_model: IsolationForest | None = None
        self.univariate_models: dict[str, IsolationForest] = {}
        self.feature_names: list[str] = []
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(self, X: np.ndarray, feature_names: list[str]) -> IsolationForestDetector:
        """Fit multivariate and per-feature univariate Isolation Forest models.

        Args:
            X: Training feature array of shape ``(n_samples, n_features)``.
            feature_names: Names corresponding to each feature column.

        Returns:
            ``self`` for method chaining.
        """
        self.feature_names = feature_names

        self.multivariate_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.multivariate_model.fit(X)

        for i, name in enumerate(feature_names):
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            model.fit(X[:, i : i + 1])
            self.univariate_models[name] = model

        logger.info(
            "Fitted IsolationForest on %d features, %d samples", len(feature_names), len(X)
        )
        
        raw_scores = -self.multivariate_model.score_samples(X)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies: 1 = anomaly, 0 = normal.

        Args:
            X: Feature array.

        Returns:
            Binary prediction array.
        """
        predictions = self.multivariate_model.predict(X)
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute normalised anomaly scores in [0, 1] (higher = more anomalous).
        Uses parameters calibrated during training to prevent data leakage.

        Args:
            X: Feature array.

        Returns:
            Score array of shape ``(n_samples,)``.
        """
        raw_scores = -self.multivariate_model.score_samples(X)
        score_range = (self._score_max - self._score_min) + 1e-10
        return np.clip((raw_scores - self._score_min) / score_range, 0.0, 1.0)

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-feature anomaly scores for root-cause analysis.

        Args:
            X: Feature array.

        Returns:
            Dict mapping feature name to normalised score array.
        """
        contributions: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.feature_names):
            model = self.univariate_models[name]
            raw_scores = -model.score_samples(X[:, i : i + 1])
            score_range = raw_scores.max() - raw_scores.min() + 1e-10
            contributions[name] = (raw_scores - raw_scores.min()) / score_range
        return contributions

    def save(self, path: str) -> None:
        """Persist models to disk.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "multivariate": self.multivariate_model,
                "univariate": self.univariate_models,
                "feature_names": self.feature_names,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "config": {
                    "contamination": self.contamination,
                    "n_estimators": self.n_estimators,
                },
            },
            path,
        )
        logger.info("Saved IsolationForest models to %s", path)

    @classmethod
    def load(cls, path: str) -> IsolationForestDetector:
        """Load persisted models from disk.

        Args:
            path: File path saved by :meth:`save`.

        Returns:
            Restored detector instance.
        """
        data = joblib.load(path)
        detector = cls(
            contamination=data["config"]["contamination"],
            n_estimators=data["config"]["n_estimators"],
        )
        detector.multivariate_model = data["multivariate"]
        detector.univariate_models = data["univariate"]
        detector.feature_names = data["feature_names"]
        detector._score_min = data.get("score_min", 0.0)
        detector._score_max = data.get("score_max", 1.0)
        return detector
