"""Tests for synthetic IoT data generator."""

import numpy as np
import pandas as pd
import pytest

from src.data.generator import SENSOR_CONFIGS, IoTDataGenerator, SensorConfig


@pytest.fixture()
def generator() -> IoTDataGenerator:
    return IoTDataGenerator(seed=42)


class TestSensorConfig:
    def test_default_noise_std(self) -> None:
        cfg = SensorConfig("test", 0.0, 1.0)
        assert cfg.noise_std == 0.02

    def test_custom_noise_std(self) -> None:
        cfg = SensorConfig("test", 0.0, 1.0, noise_std=0.5)
        assert cfg.noise_std == 0.5

    def test_all_sensor_types_defined(self) -> None:
        expected = {
            "pump_1_speed", "pump_2_speed", "valve_1_position", "valve_2_position", "dosing_rate",
            "pump_flow", "inlet_flow", "outlet_flow", "recycle_flow",
            "pressure", "inlet_pressure", "pump_pressure", "filter_pressure",
            "tank_1_level", "tank_2_level", "tank_3_level", "reservoir_level",
            "temperature", "pump_temperature", "motor_temperature",
            "pH", "conductivity", "turbidity", "chlorine",
            "power", "motor_current", "motor_voltage", "vibration",
        }
        assert set(SENSOR_CONFIGS.keys()) == expected
        assert {"temperature", "vibration", "pressure", "power"}.issubset(set(SENSOR_CONFIGS.keys()))


class TestBaseSignal:
    def test_output_length(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        signal = generator._generate_base_signal(config, 100)
        assert len(signal) == 100

    def test_values_within_reasonable_range(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        signal = generator._generate_base_signal(config, 1000)
        # With noise, values should roughly be within range ± some margin
        assert signal.min() > config.normal_min - 10 * config.noise_std
        assert signal.max() < config.normal_max + 10 * config.noise_std


class TestAnomalyInjection:
    def test_point_anomaly_deviates(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        data = generator._generate_base_signal(config, 100)
        original = data[50]
        anomalous = generator._inject_point_anomaly(data, 50, config)
        assert anomalous != original

    def test_contextual_anomaly_swaps(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        data = np.arange(100, dtype=float)
        result = generator._inject_contextual_anomaly(data, 10, config)
        assert result == data[60]  # idx + len//2

    def test_collective_anomaly_affects_range(self, generator: IoTDataGenerator) -> None:
        config = SENSOR_CONFIGS["temperature"]
        data = np.ones(100) * 70.0
        original = data.copy()
        generator._inject_collective_anomaly(data, 20, 10, config)
        # At least some points in the range should differ from original
        assert not np.array_equal(data[20:30], original[20:30])


class TestGenerate:
    def test_row_count_1_day(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1, sensors_per_type=1)
        expected = 1 * 24 * 60 * len(SENSOR_CONFIGS)
        assert len(df) == expected

    def test_columns_present(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        expected_cols = {"timestamp", "sensor_type", "sensor_id", "value", "is_anomaly"}
        assert set(df.columns) == expected_cols

    def test_anomaly_rate_approximate(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=7, anomaly_rate=0.05)
        actual_rate = df["is_anomaly"].mean()
        assert actual_rate > 0.01

    def test_sensor_ids_format(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1, sensors_per_type=2)
        ids = df["sensor_id"].unique()
        assert "temperature_00" in ids
        assert "temperature_01" in ids

    def test_timestamps_sequential(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        temp_df = df[df["sensor_id"] == "temperature_00"]
        timestamps = pd.to_datetime(temp_df["timestamp"])
        diffs = timestamps.diff().dropna()
        # All diffs should be exactly 1 minute
        assert (diffs == pd.Timedelta(minutes=1)).all()

    def test_reproducibility(self) -> None:
        gen1 = IoTDataGenerator(seed=123)
        gen2 = IoTDataGenerator(seed=123)
        df1 = gen1.generate(days=1)
        df2 = gen2.generate(days=1)
        np.testing.assert_array_equal(df1["value"].values, df2["value"].values)
        np.testing.assert_array_equal(df1["is_anomaly"].values, df2["is_anomaly"].values)
        np.testing.assert_array_equal(df1["sensor_id"].values, df2["sensor_id"].values)

    def test_multiple_sensors_per_type(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1, sensors_per_type=3)
        for sensor_type in SENSOR_CONFIGS:
            type_ids = df[df["sensor_type"] == sensor_type]["sensor_id"].nunique()
            assert type_ids == 3


class TestSaveToCsv:
    def test_save_creates_file(self, generator: IoTDataGenerator, tmp_path) -> None:
        df = generator.generate(days=1)
        out = tmp_path / "output.csv"
        generator.save_to_csv(df, str(out))
        assert out.exists()

    def test_save_creates_parent_dirs(self, generator: IoTDataGenerator, tmp_path) -> None:
        df = generator.generate(days=1)
        out = tmp_path / "a" / "b" / "output.csv"
        generator.save_to_csv(df, str(out))
        assert out.exists()

    def test_saved_csv_roundtrip(self, generator: IoTDataGenerator, tmp_path) -> None:
        df = generator.generate(days=1)
        out = tmp_path / "output.csv"
        generator.save_to_csv(df, str(out))
        loaded = pd.read_csv(str(out))
        assert len(loaded) == len(df)
        assert set(loaded.columns) == set(df.columns)


class TestIndustrialProcessGenerator:
    """Focused Phase 2A tests for the realistic correlated industrial benchmark."""

    def test_deterministic_generation(self) -> None:
        gen1 = IoTDataGenerator(seed=999)
        gen2 = IoTDataGenerator(seed=999)
        df1 = gen1.generate(days=1)
        df2 = gen2.generate(days=1)
        np.testing.assert_array_equal(df1["value"].values, df2["value"].values)
        np.testing.assert_array_equal(df1["is_anomaly"].values, df2["is_anomaly"].values)
        assert [m.to_dict() for m in gen1.anomaly_metadata] == [m.to_dict() for m in gen2.anomaly_metadata]

    def test_expected_feature_names(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        sensor_types = set(df["sensor_type"].unique())
        assert sensor_types == set(SENSOR_CONFIGS.keys())

    def test_expected_feature_count(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        assert df["sensor_id"].nunique() == 28

    def test_chronological_timestamps(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        temp_df = df[df["sensor_id"] == "pump_flow_00"]
        ts = pd.to_datetime(temp_df["timestamp"])
        assert ts.is_monotonic_increasing
        assert len(ts) == len(ts.drop_duplicates())

    def test_anomaly_metadata(self, generator: IoTDataGenerator) -> None:
        generator.generate(days=1)
        assert len(generator.anomaly_metadata) >= 5
        for meta in generator.anomaly_metadata:
            assert meta.anomaly_id > 0
            assert meta.anomaly_type in [
                "sudden_spike", "sudden_drop", "gradual_drift", "sensor_bias",
                "increased_noise", "sensor_dropout", "stuck_sensor", "oscillation",
                "correlated_multi_sensor", "gradual_process_degradation"
            ]
            assert meta.start_idx < meta.end_idx
            assert len(meta.affected_sensors) > 0
            assert meta.severity > 0

    def test_anomaly_windows(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        stats = generator.validate_dataset(df)
        assert stats["anomaly_windows_count"] >= 5
        assert stats["min_anomaly_duration"] >= 3
        assert stats["max_anomaly_duration"] <= 50

    def test_gradual_drift(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        drift_metas = [m for m in generator.anomaly_metadata if m.anomaly_type == "gradual_drift"]
        assert len(drift_metas) >= 1
        meta = drift_metas[0]
        pivoted = df.pivot_table(index="timestamp", columns="sensor_id", values="value")
        target_col = f"{meta.affected_sensors[0]}_00"
        drift_values = pivoted[target_col].iloc[meta.start_idx : meta.end_idx].values
        # Drift should be non-constant
        assert np.ptp(drift_values) > 0.1

    def test_correlated_anomaly(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        corr_metas = [m for m in generator.anomaly_metadata if m.anomaly_type == "correlated_multi_sensor"]
        assert len(corr_metas) >= 1
        meta = corr_metas[0]
        assert len(meta.affected_sensors) >= 2

    def test_both_classes_in_validation(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        stats = generator.validate_dataset(df)
        assert stats["val_has_both_classes"] is True
        assert stats["val_anomalies"] > 0
        assert stats["val_size"] - stats["val_anomalies"] > 0

    def test_both_classes_in_test(self, generator: IoTDataGenerator) -> None:
        df = generator.generate(days=1)
        stats = generator.validate_dataset(df)
        assert stats["test_has_both_classes"] is True
        assert stats["test_anomalies"] > 0
        assert stats["test_size"] - stats["test_anomalies"] > 0

    def test_missing_value_handling(self) -> None:
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor(scaler_type="standard")
        # DataFrame with NaN
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
            "sensor_1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "sensor_2": [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "is_anomaly": [0] * 10,
        })
        X_scaled, _ = prep.fit_transform(df)
        assert not np.isnan(X_scaled).any()
        # Transform unseen data with NaNs
        df_test = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-02", periods=5, freq="min"),
            "sensor_1": [np.nan, 2.0, 3.0, 4.0, 5.0],
            "sensor_2": [10.0, np.nan, 30.0, 40.0, 50.0],
        })
        X_test_scaled = prep.transform(df_test)
        assert not np.isnan(X_test_scaled).any()

    def test_preprocessing_compatibility(self, generator: IoTDataGenerator) -> None:
        from src.data.preprocessor import DataPreprocessor
        df = generator.generate(days=1)
        prep = DataPreprocessor(scaler_type="standard")
        pivoted = prep.pivot_sensors(df)
        X_scaled, y = prep.fit_transform(pivoted)
        assert X_scaled.shape == (1440, 28)
        assert y is not None and len(y) == 1440
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = prep.split_data(X_scaled, y)
        assert len(X_tr) == 1007
        assert len(X_va) == 217
        assert len(X_te) == 216
        assert len(np.unique(y_va)) == 2
        assert len(np.unique(y_te)) == 2
