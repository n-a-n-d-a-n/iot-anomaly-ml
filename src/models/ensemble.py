"""Ensemble anomaly detector combining multiple scoring methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EnsembleResult:
    """Output of :meth:`EnsembleDetector.predict`.

    Attributes:
        scores: Weighted ensemble anomaly scores.
        predictions: Binary predictions based on threshold.
        method_scores: Per-method score arrays.
        feature_contributions: Aggregated per-feature contribution arrays.
    """

    scores: np.ndarray
    predictions: np.ndarray
    method_scores: dict[str, np.ndarray] = field(default_factory=dict)
    feature_contributions: dict[str, np.ndarray] = field(default_factory=dict)


class EnsembleDetector:
    """Weighted ensemble of anomaly detection models.

    Args:
        weights: Mapping of model name to weight. Defaults to equal
            weights for ``isolation_forest``, ``autoencoder``, ``dbscan``.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights: dict[str, float] = weights or {
            "isolation_forest": 0.25,
            "autoencoder": 0.25,
            "dbscan": 0.25,
            "gmm": 0.25,
        }
        
        if not all(isinstance(v, (int, float)) for v in self.weights.values()):
            raise ValueError("All weights must be numeric")
        if any(v < 0 for v in self.weights.values()):
            raise ValueError("Weights cannot be negative")
            
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Total weight must be positive")
            
        self.weights = {k: float(v) / total for k, v in self.weights.items()}
        self.models: dict[str, Any] = {}
        self.threshold: float = 0.5

    def add_model(self, name: str, model: Any) -> None:
        """Register a trained model with the ensemble.

        Args:
            name: Key identifying the model (must match a key in *weights*).
            model: Any object implementing ``score_samples`` and
                ``get_feature_contributions``.
        """
        self.models[name] = model

    def calibrate_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_precision: float = 0.8,
    ) -> float:
        """Find the threshold that maximises F1 on a labelled validation set.

        Args:
            X: Validation features.
            y: Ground-truth binary labels.
            target_precision: Informational; the search actually maximises F1.

        Returns:
            Optimal threshold value.
        """
        scores = self._compute_ensemble_scores(X)

        best_threshold = 0.5
        best_f1 = 0.0

        for thresh in np.linspace(0, 1, 100):
            preds = (scores >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y == 1)))
            fp = int(np.sum((preds == 1) & (y == 0)))
            fn = int(np.sum((preds == 0) & (y == 1)))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresh)

        self.threshold = best_threshold
        logger.info("Calibrated threshold=%.4f  best-F1=%.4f", best_threshold, best_f1)
        return best_threshold

    def _compute_ensemble_scores(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of per-model anomaly scores.

        Args:
            X: Feature array.

        Returns:
            Ensemble score array in [0, 1].
        """
        total_weight = sum(self.weights.get(name, 0.0) for name in self.models)
        ensemble_scores = np.zeros(len(X))

        if total_weight > 0:
            for name, model in self.models.items():
                weight = self.weights.get(name, 0.0)
                ensemble_scores += weight * model.score_samples(X) / total_weight

        return ensemble_scores

    def predict(self, X: np.ndarray, compute_contributions: bool = True) -> EnsembleResult:
        """Score and classify samples using all registered models.

        Args:
            X: Feature array.
            compute_contributions: Whether to compute per-feature contributions. Defaults to True.

        Returns:
            :class:`EnsembleResult` with scores, predictions, per-method
            breakdowns, and aggregated feature contributions.
        """
        method_scores = {name: model.score_samples(X) for name, model in self.models.items()}
        total_weight = sum(self.weights.get(name, 0.0) for name in self.models)
        ensemble_scores = np.zeros(len(X))
        if total_weight > 0:
            for name, scores in method_scores.items():
                weight = self.weights.get(name, 0.0)
                ensemble_scores += weight * scores / total_weight

        predictions = (ensemble_scores >= self.threshold).astype(int)

        feature_contributions: dict[str, np.ndarray] = {}
        if compute_contributions:
            all_contributions: dict[str, list[np.ndarray]] = {}
            for name, model in self.models.items():
                contribs = model.get_feature_contributions(X)
                for feat, vals in contribs.items():
                    if feat not in all_contributions:
                        all_contributions[feat] = []
                    all_contributions[feat].append(vals * self.weights.get(name, 0))

            feature_contributions = {
                feat: np.mean(vals, axis=0) for feat, vals in all_contributions.items()
            }

        return EnsembleResult(
            scores=ensemble_scores,
            predictions=predictions,
            method_scores=method_scores,
            feature_contributions=feature_contributions,
        )
