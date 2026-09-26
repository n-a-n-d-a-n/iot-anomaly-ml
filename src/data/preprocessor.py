"""Data preprocessing pipeline: normalization, windowing, and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class WindowedData:
    """Container for windowed time-series data.

    Attributes:
        X: Sliding-window feature tensor of shape ``(n_samples, window_size, n_features)``.
        y: Ground-truth labels for the last point in each window, shape ``(n_samples,)``.
        timestamps: Timestamp of the last point in each window.
        feature_names: Column names corresponding to the feature dimension.
    """

    X: np.ndarray
    y: np.ndarray | None
    timestamps: np.ndarray
    feature_names: list[str] = field(default_factory=list)


class DataPreprocessor:
    """Preprocessing pipeline for IoT sensor data.

    Handles pivot from long to wide format, scaling, windowing, and
    statistical feature extraction.

    Args:
        scaler_type: ``"standard"`` for zero-mean/unit-variance or ``"minmax"``
            for [0, 1] scaling.
    """

    def __init__(self, scaler_type: str = "standard") -> None:
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.fitted: bool = False
        self.feature_names: list[str] = []
        self.medians: np.ndarray | None = None

    def pivot_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-format sensor data to wide format (one column per sensor).

        Args:
            df: Long-format DataFrame with ``timestamp``, ``sensor_id``,
                ``value``, and ``is_anomaly`` columns.

        Returns:
            Wide-format DataFrame indexed by timestamp with one column per
            sensor plus an ``is_anomaly`` column.
        """
        pivoted = df.pivot_table(
            index="timestamp",
            columns="sensor_id",
            values="value",
            aggfunc="first",
        ).reset_index()

        labels = df.groupby("timestamp")["is_anomaly"].max().reset_index()
        pivoted = pivoted.merge(labels, on="timestamp")
        return pivoted

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Fit the scaler on *df* and return transformed features and labels.

        Args:
            df: Wide-format DataFrame (from :meth:`pivot_sensors`).

        Returns:
            Tuple of ``(X_scaled, y)`` where *y* is ``None`` when the
            ``is_anomaly`` column is absent.
        """
        feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
        X = df[feature_cols].values.astype(float)
        y = df["is_anomaly"].values if "is_anomaly" in df.columns else None

        # Compute and store medians on training data for missing-value imputation
        self.medians = np.nanmedian(X, axis=0)
        self.medians = np.nan_to_num(self.medians, nan=0.0)

        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = np.where(nan_mask, self.medians, X)

        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        self.feature_names = feature_cols
        logger.info("Fitted scaler on %d features, %d samples", len(feature_cols), len(X))
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using the already-fitted scaler.

        Args:
            df: Wide-format DataFrame with the same feature columns used during fit.

        Returns:
            Scaled feature array.

        Raises:
            ValueError: If the scaler has not been fitted yet.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
        X = df[feature_cols].values.astype(float)

        nan_mask = np.isnan(X)
        if nan_mask.any() and self.medians is not None:
            X = np.where(nan_mask, self.medians, X)

        return self.scaler.transform(X)

    def create_windows(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        timestamps: np.ndarray,
        window_size: int = 60,
    ) -> WindowedData:
        """Create sliding windows for temporal context.

        Args:
            X: Feature array of shape ``(n_total, n_features)``.
            y: Optional label array.
            timestamps: Timestamp array aligned with *X*.
            window_size: Number of time steps per window.

        Returns:
            :class:`WindowedData` with tensors of shape
            ``(n_total - window_size + 1, window_size, n_features)``.
        """
        n_samples = len(X) - window_size + 1
        n_features = X.shape[1]

        X_windowed = np.zeros((n_samples, window_size, n_features))
        y_windowed = np.zeros(n_samples) if y is not None else None
        ts_windowed = timestamps[window_size - 1 :]

        for i in range(n_samples):
            X_windowed[i] = X[i : i + window_size]
            if y is not None:
                y_windowed[i] = y[i + window_size - 1]

        return WindowedData(X_windowed, y_windowed, ts_windowed, self.feature_names)

    def extract_window_features(self, windowed: WindowedData) -> np.ndarray:
        """Extract statistical features from each window for non-sequential models.

        Per window and per feature the following statistics are computed:
        mean, std, min, max, and linear trend (slope).

        Args:
            windowed: Output from :meth:`create_windows`.

        Returns:
            Array of shape ``(n_samples, n_features * 5)``.
        """
        n_samples, window_size, n_features = windowed.X.shape

        features: list[list[float]] = []
        for i in range(n_samples):
            window = windowed.X[i]
            feat: list[float] = []
            for j in range(n_features):
                col = window[:, j]
                feat.extend(
                    [
                        float(np.mean(col)),
                        float(np.std(col)),
                        float(np.min(col)),
                        float(np.max(col)),
                        float(np.polyfit(range(len(col)), col, 1)[0]),
                    ]
                )
            features.append(feat)

        return np.array(features)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Split data into train / validation / test sets preserving temporal order.

        Args:
            X: Feature array.
            y: Label array.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.

        Returns:
            Three ``(X, y)`` tuples for train, validation, and test.
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:val_end], y[train_end:val_end]),
            (X[val_end:], y[val_end:]),
        )
