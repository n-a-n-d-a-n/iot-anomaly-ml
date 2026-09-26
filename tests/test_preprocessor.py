"""Tests for data preprocessing and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import IoTDataGenerator
from src.data.preprocessor import DataPreprocessor


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Generate a small sample dataset."""
    gen = IoTDataGenerator(seed=0)
    return gen.generate(days=1, sensors_per_type=1)


@pytest.fixture()
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor(scaler_type="standard")


@pytest.fixture()
def pivoted_df(preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> pd.DataFrame:
    return preprocessor.pivot_sensors(sample_df)


class TestPivotSensors:
    def test_output_has_sensor_columns(self, pivoted_df: pd.DataFrame) -> None:
        assert "temperature_00" in pivoted_df.columns
        assert "vibration_00" in pivoted_df.columns
        assert "pressure_00" in pivoted_df.columns
        assert "power_00" in pivoted_df.columns

    def test_output_has_timestamp(self, pivoted_df: pd.DataFrame) -> None:
        assert "timestamp" in pivoted_df.columns

    def test_output_has_anomaly_label(self, pivoted_df: pd.DataFrame) -> None:
        assert "is_anomaly" in pivoted_df.columns

    def test_row_count_matches_timestamps(
        self, sample_df: pd.DataFrame, pivoted_df: pd.DataFrame
    ) -> None:
        unique_ts = sample_df["timestamp"].nunique()
        assert len(pivoted_df) == unique_ts


class TestFitTransform:
    def test_output_shapes(self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame) -> None:
        X, y = preprocessor.fit_transform(pivoted_df)
        n_features = len([c for c in pivoted_df.columns if c not in ("timestamp", "is_anomaly")])
        assert X.shape == (len(pivoted_df), n_features)
        assert y.shape == (len(pivoted_df),)

    def test_standard_scaler_normalizes(
        self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame
    ) -> None:
        X, _ = preprocessor.fit_transform(pivoted_df)
        # Mean should be near 0 for StandardScaler
        assert abs(X.mean()) < 0.1

    def test_sets_fitted_flag(
        self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame
    ) -> None:
        preprocessor.fit_transform(pivoted_df)
        assert preprocessor.fitted is True

    def test_stores_feature_names(
        self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame
    ) -> None:
        preprocessor.fit_transform(pivoted_df)
        assert len(preprocessor.feature_names) == 28


class TestTransform:
    def test_transform_without_fit_raises(self, preprocessor: DataPreprocessor) -> None:
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError, match="fitted"):
            preprocessor.transform(df)

    def test_transform_after_fit(
        self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame
    ) -> None:
        preprocessor.fit_transform(pivoted_df)
        X = preprocessor.transform(pivoted_df)
        assert X.shape[0] == len(pivoted_df)


class TestMinMaxScaler:
    def test_minmax_scales_to_unit_range(self, pivoted_df: pd.DataFrame) -> None:
        pp = DataPreprocessor(scaler_type="minmax")
        X, _ = pp.fit_transform(pivoted_df)
        assert X.min() >= -0.01
        assert X.max() <= 1.01


class TestCreateWindows:
    def test_window_shape(self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame) -> None:
        X, y = preprocessor.fit_transform(pivoted_df)
        ts = pivoted_df["timestamp"].values
        windowed = preprocessor.create_windows(X, y, ts, window_size=10)
        expected_samples = len(X) - 10 + 1
        assert windowed.X.shape == (expected_samples, 10, X.shape[1])

    def test_labels_from_last_point(
        self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame
    ) -> None:
        X, y = preprocessor.fit_transform(pivoted_df)
        ts = pivoted_df["timestamp"].values
        windowed = preprocessor.create_windows(X, y, ts, window_size=5)
        # Label of window i should be y[i + window_size - 1]
        for i in range(min(10, len(windowed.y))):
            assert windowed.y[i] == y[i + 4]

    def test_timestamps_aligned(
        self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame
    ) -> None:
        X, y = preprocessor.fit_transform(pivoted_df)
        ts = pivoted_df["timestamp"].values
        windowed = preprocessor.create_windows(X, y, ts, window_size=10)
        assert len(windowed.timestamps) == windowed.X.shape[0]


class TestExtractWindowFeatures:
    def test_feature_count(self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame) -> None:
        X, y = preprocessor.fit_transform(pivoted_df)
        ts = pivoted_df["timestamp"].values
        windowed = preprocessor.create_windows(X, y, ts, window_size=10)
        features = preprocessor.extract_window_features(windowed)
        # 5 stats per feature: mean, std, min, max, trend
        assert features.shape == (windowed.X.shape[0], X.shape[1] * 5)

    def test_no_nan_values(self, preprocessor: DataPreprocessor, pivoted_df: pd.DataFrame) -> None:
        X, y = preprocessor.fit_transform(pivoted_df)
        ts = pivoted_df["timestamp"].values
        windowed = preprocessor.create_windows(X, y, ts, window_size=10)
        features = preprocessor.extract_window_features(windowed)
        assert not np.isnan(features).any()


class TestSplitData:
    def test_split_proportions(self, preprocessor: DataPreprocessor) -> None:
        X = np.random.randn(1000, 4)
        y = np.random.randint(0, 2, 1000)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(
            X, y, train_ratio=0.7, val_ratio=0.15
        )
        assert len(X_train) == 700
        assert len(X_val) == 150
        assert len(X_test) == 150

    def test_temporal_order_preserved(self, preprocessor: DataPreprocessor) -> None:
        X = np.arange(100).reshape(-1, 1)
        y = np.zeros(100)
        (X_train, _), (X_val, _), (X_test, _) = preprocessor.split_data(X, y)
        # Train should contain the first items, test the last
        assert X_train[0, 0] == 0
        assert X_test[-1, 0] == 99
        assert X_val[0, 0] > X_train[-1, 0]
