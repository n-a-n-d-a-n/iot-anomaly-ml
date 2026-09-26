"""Synthetic Industrial IoT sensor data generator with realistic process correlation.

Simulates a physical multi-stage water treatment / industrial pumping plant with
28 correlated process, electrical, thermal, flow, pressure, level, and quality variables.
Includes smooth operating regimes (Low, Normal, High), realistic temporal dynamics,
and controlled contiguous anomaly window injections across 10 industrial anomaly types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SensorConfig:
    """Configuration for an industrial IoT sensor.

    Attributes:
        name: Unique sensor identifier.
        normal_min: Lower bound of the normal operating envelope.
        normal_max: Upper bound of the normal operating envelope.
        noise_std: Standard deviation of physical/measurement noise.
        group: Sensor domain group (flow, pressure, level, temp, quality, electrical, actuator).
        unit: Engineering unit of measurement.
        description: Functional description of the sensor's role in the process.
    """

    name: str
    normal_min: float
    normal_max: float
    noise_std: float = 0.02
    group: str = "general"
    unit: str = ""
    description: str = ""


@dataclass
class AnomalyMetadata:
    """Metadata describing an injected anomaly event window.

    Attributes:
        anomaly_id: Unique sequential identifier.
        anomaly_type: Category of anomaly (e.g. sudden_spike, gradual_drift).
        start_time: ISO timestamp of anomaly onset.
        end_time: ISO timestamp of anomaly conclusion.
        start_idx: Time-step index of anomaly onset.
        end_idx: Time-step index of anomaly conclusion (exclusive).
        affected_sensors: List of sensors perturbed by this anomaly.
        severity: Magnitude scaling factor.
    """

    anomaly_id: int
    anomaly_type: str
    start_time: str
    end_time: str
    start_idx: int
    end_idx: int
    affected_sensors: list[str]
    severity: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "affected_sensors": self.affected_sensors,
            "severity": self.severity,
        }


# Comprehensive industrial process sensor configuration (28 variables across 7 groups)
SENSOR_CONFIGS: dict[str, SensorConfig] = {
    # 1. Actuator / Control Variables (5)
    "pump_1_speed": SensorConfig(
        "pump_1_speed", 1000.0, 2000.0, noise_std=8.0, group="actuator", unit="rpm", description="Primary pump rotational speed"
    ),
    "pump_2_speed": SensorConfig(
        "pump_2_speed", 800.0, 1800.0, noise_std=7.0, group="actuator", unit="rpm", description="Secondary pump rotational speed"
    ),
    "valve_1_position": SensorConfig(
        "valve_1_position", 20.0, 100.0, noise_std=0.3, group="actuator", unit="%", description="Inlet control valve opening"
    ),
    "valve_2_position": SensorConfig(
        "valve_2_position", 20.0, 100.0, noise_std=0.3, group="actuator", unit="%", description="Discharge control valve opening"
    ),
    "dosing_rate": SensorConfig(
        "dosing_rate", 5.0, 25.0, noise_std=0.1, group="actuator", unit="L/h", description="Chemical treatment dosing pump rate"
    ),

    # 2. Flow Sensors (4)
    "pump_flow": SensorConfig(
        "pump_flow", 30.0, 90.0, noise_std=0.4, group="flow", unit="m3/h", description="Discharge volumetric flow from primary pump"
    ),
    "inlet_flow": SensorConfig(
        "inlet_flow", 35.0, 95.0, noise_std=0.5, group="flow", unit="m3/h", description="Raw water inlet flow rate"
    ),
    "outlet_flow": SensorConfig(
        "outlet_flow", 25.0, 80.0, noise_std=0.4, group="flow", unit="m3/h", description="Treated water discharge flow rate"
    ),
    "recycle_flow": SensorConfig(
        "recycle_flow", 2.0, 25.0, noise_std=0.25, group="flow", unit="m3/h", description="System recirculation flow rate"
    ),

    # 3. Pressure Sensors (4)
    "pressure": SensorConfig(
        "pressure", 1.5, 5.5, noise_std=0.03, group="pressure", unit="bar", description="Main process header pressure"
    ),
    "inlet_pressure": SensorConfig(
        "inlet_pressure", 1.5, 4.5, noise_std=0.025, group="pressure", unit="bar", description="Raw water feed header pressure"
    ),
    "pump_pressure": SensorConfig(
        "pump_pressure", 2.0, 6.5, noise_std=0.035, group="pressure", unit="bar", description="Pump discharge head pressure"
    ),
    "filter_pressure": SensorConfig(
        "filter_pressure", 0.5, 3.5, noise_std=0.02, group="pressure", unit="bar", description="Pressure drop across filtration unit"
    ),

    # 4. Level Sensors (4)
    "tank_1_level": SensorConfig(
        "tank_1_level", 2.0, 8.0, noise_std=0.03, group="level", unit="m", description="Primary buffer tank liquid level"
    ),
    "tank_2_level": SensorConfig(
        "tank_2_level", 2.0, 7.5, noise_std=0.03, group="level", unit="m", description="Coagulation/settling tank liquid level"
    ),
    "tank_3_level": SensorConfig(
        "tank_3_level", 1.5, 6.5, noise_std=0.025, group="level", unit="m", description="Treated water storage tank liquid level"
    ),
    "reservoir_level": SensorConfig(
        "reservoir_level", 5.0, 10.0, noise_std=0.04, group="level", unit="m", description="Raw supply reservoir level"
    ),

    # 5. Temperature Sensors (3)
    "temperature": SensorConfig(
        "temperature", 15.0, 40.0, noise_std=0.12, group="temperature", unit="celsius", description="Bulk process fluid temperature"
    ),
    "pump_temperature": SensorConfig(
        "pump_temperature", 40.0, 85.0, noise_std=0.25, group="temperature", unit="celsius", description="Pump casing temperature"
    ),
    "motor_temperature": SensorConfig(
        "motor_temperature", 50.0, 100.0, noise_std=0.30, group="temperature", unit="celsius", description="Electric motor stator winding temperature"
    ),

    # 6. Quality Sensors (4)
    "pH": SensorConfig(
        "pH", 6.0, 8.5, noise_std=0.02, group="quality", unit="pH", description="Treated water acidity/alkalinity"
    ),
    "conductivity": SensorConfig(
        "conductivity", 400.0, 900.0, noise_std=1.5, group="quality", unit="uS/cm", description="Electrical conductivity measuring dissolved ionic solids"
    ),
    "turbidity": SensorConfig(
        "turbidity", 0.5, 6.0, noise_std=0.03, group="quality", unit="NTU", description="Nephelometric turbidity of process water"
    ),
    "chlorine": SensorConfig(
        "chlorine", 0.5, 3.5, noise_std=0.015, group="quality", unit="mg/L", description="Free active chlorine residual"
    ),

    # 7. Electrical / Motor Sensors (4)
    "power": SensorConfig(
        "power", 10.0, 40.0, noise_std=0.15, group="electrical", unit="kW", description="Total real electrical power consumption"
    ),
    "motor_current": SensorConfig(
        "motor_current", 15.0, 55.0, noise_std=0.25, group="electrical", unit="A", description="Primary drive motor RMS phase current"
    ),
    "motor_voltage": SensorConfig(
        "motor_voltage", 380.0, 420.0, noise_std=0.6, group="electrical", unit="V", description="3-phase bus line-to-line RMS voltage"
    ),
    "vibration": SensorConfig(
        "vibration", 0.5, 6.0, noise_std=0.035, group="electrical", unit="mm/s", description="Drive bearing peak-to-peak vibration velocity"
    ),
}

AnomalyType = Literal[
    "point",
    "contextual",
    "collective",
    "sudden_spike",
    "sudden_drop",
    "gradual_drift",
    "sensor_bias",
    "increased_noise",
    "sensor_dropout",
    "stuck_sensor",
    "oscillation",
    "correlated_multi_sensor",
    "gradual_process_degradation",
]


class IoTDataGenerator:
    """Generate synthetic Industrial IoT process sensor readings with correlated physics.

    Args:
        seed: Random seed for deterministic reproducibility.
        sensor_configs: Optional custom sensor configuration mapping. Defaults to SENSOR_CONFIGS.
    """

    def __init__(
        self,
        seed: int = 42,
        sensor_configs: dict[str, SensorConfig] | None = None,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sensor_configs = sensor_configs or SENSOR_CONFIGS
        self.anomaly_metadata: list[AnomalyMetadata] = []

    def _generate_base_signal(self, config: SensorConfig, n_points: int) -> np.ndarray:
        """Legacy helper: Generate normal sensor readings with slight temporal correlation.

        Args:
            config: Sensor configuration with range and noise parameters.
            n_points: Number of data points to generate.

        Returns:
            Array of simulated sensor values.
        """
        mean = (config.normal_min + config.normal_max) / 2
        amplitude = (config.normal_max - config.normal_min) / 4
        t = np.linspace(0, 4 * np.pi, n_points)
        signal = mean + amplitude * np.sin(t + self.rng.random() * np.pi)
        noise = self.rng.normal(0, config.noise_std, n_points)
        return signal + noise

    def _inject_point_anomaly(self, data: np.ndarray, idx: int, config: SensorConfig) -> float:
        """Legacy helper: Inject a sudden spike or drop at a single index."""
        direction = self.rng.choice([-1, 1])
        magnitude = (config.normal_max - config.normal_min) * self.rng.uniform(0.5, 1.5)
        return float(data[idx] + direction * magnitude)

    def _inject_contextual_anomaly(
        self, data: np.ndarray, idx: int, config: SensorConfig
    ) -> float:
        """Legacy helper: Inject a contextual anomaly (normal value at the wrong time)."""
        distant_idx = (idx + len(data) // 2) % len(data)
        return float(data[distant_idx])

    def _inject_collective_anomaly(
        self,
        data: np.ndarray,
        start_idx: int,
        duration: int,
        config: SensorConfig,
    ) -> np.ndarray:
        """Legacy helper: Inject a gradual drift anomaly over a range of indices."""
        drift = np.linspace(0, (config.normal_max - config.normal_min) * 0.8, duration)
        data[start_idx : start_idx + duration] += drift * self.rng.choice([-1, 1])
        return data

    def _generate_regime_signal(self, n_points: int) -> np.ndarray:
        """Generate smooth operating regimes: LOW (0.85), NORMAL (1.00), HIGH (1.15).

        Transitions are smoothed with a rolling filter to prevent unrealistic instantaneous jumps.
        """
        regime_cycle = [1.0, 1.15, 1.0, 0.85]  # NORMAL -> HIGH -> NORMAL -> LOW
        block_len = max(180, n_points // 8)
        raw_regimes: list[float] = []
        cycle_idx = 0
        while len(raw_regimes) < n_points:
            target = regime_cycle[cycle_idx % len(regime_cycle)]
            raw_regimes.extend([target] * block_len)
            cycle_idx += 1
        raw_regimes_arr = np.array(raw_regimes[:n_points], dtype=float)

        # Smooth transitions using moving average
        kernel_size = min(31, n_points // 10 if n_points >= 10 else 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(raw_regimes_arr, kernel, mode="same")
        return smoothed

    def _generate_ar1_noise(self, n_points: int, std: float, phi: float = 0.7) -> np.ndarray:
        """Generate autoregressive AR(1) colored noise for temporal continuity."""
        innovations = self.rng.normal(0, std * np.sqrt(1.0 - phi**2), n_points)
        noise = np.zeros(n_points)
        noise[0] = innovations[0]
        for t in range(1, n_points):
            noise[t] = phi * noise[t - 1] + innovations[t]
        return noise

    def _simulate_process(
        self, n_points: int, sampling_interval_minutes: int
    ) -> dict[str, np.ndarray]:
        """Simulate the coupled physical and control equations for all 28 variables."""
        regime = self._generate_regime_signal(n_points)
        t = np.arange(n_points)
        diurnal_period = (24 * 60) // sampling_interval_minutes
        diurnal = np.sin(2 * np.pi * t / max(1, diurnal_period))
        slow_trend = 0.4 * np.sin(2 * np.pi * t / max(1, 7 * diurnal_period))

        # Generate AR(1) correlated noise for each sensor
        noise = {
            k: self._generate_ar1_noise(n_points, cfg.noise_std)
            for k, cfg in self.sensor_configs.items()
        }

        # 1. Actuators
        pump_1_speed = 1500.0 * regime + 35.0 * diurnal + noise["pump_1_speed"]
        pump_2_speed = 1300.0 * regime + 25.0 * diurnal + noise["pump_2_speed"]
        valve_1_pos = 60.0 + 10.0 * (regime - 1.0) + noise["valve_1_position"]
        valve_2_pos = 55.0 + 8.0 * (regime - 1.0) + noise["valve_2_position"]
        dosing_rate = 12.0 * regime + 0.5 * diurnal + noise["dosing_rate"]

        # 2. Flow Sensors (physically driven by pump speed and valve position)
        pump_flow = 0.045 * pump_1_speed * (valve_1_pos / 60.0) + noise["pump_flow"]
        inlet_flow = 0.86 * pump_flow + 0.012 * pump_2_speed + noise["inlet_flow"]
        outlet_flow = 0.80 * pump_flow * (valve_2_pos / 55.0) + noise["outlet_flow"]
        recycle_flow = (
            8.0 + 0.15 * (pump_flow - outlet_flow) + 0.005 * pump_2_speed + noise["recycle_flow"]
        )

        # 3. Pressure Sensors
        pump_pressure = 1.4 + 0.0024 * pump_1_speed - 0.016 * pump_flow + noise["pump_pressure"]
        inlet_pressure = 2.1 + 0.016 * inlet_flow - 0.010 * (valve_1_pos - 60.0) + noise["inlet_pressure"]
        pressure = (
            0.55 * pump_pressure
            + 0.45 * inlet_pressure
            - 0.015 * (valve_2_pos - 55.0)
            + noise["pressure"]
        )
        filter_pressure = 0.7 + 0.024 * pump_flow + noise["filter_pressure"]

        # 4. Level Sensors
        tank_1_level = 5.0 + 0.035 * (inlet_flow - pump_flow) + slow_trend + noise["tank_1_level"]
        tank_2_level = 4.5 + 0.030 * (pump_flow - outlet_flow) + noise["tank_2_level"]
        tank_3_level = (
            4.0 + 0.025 * outlet_flow + 0.035 * (recycle_flow - 10.0) + noise["tank_3_level"]
        )
        reservoir_level = (
            8.0 - 0.022 * (inlet_flow - 60.0) + 0.018 * (recycle_flow - 10.0) + noise["reservoir_level"]
        )

        # 5. Electrical Sensors
        motor_load = 0.007 * pump_1_speed + 0.16 * pump_pressure + 0.04 * (100.0 - valve_1_pos)
        motor_current = 15.0 + 0.85 * motor_load + noise["motor_current"]
        motor_voltage = 402.0 - 0.10 * (motor_current - 30.0) + noise["motor_voltage"]
        # 3-phase real electrical power: P = sqrt(3) * V * I * pf / 1000
        power = (np.sqrt(3.0) * motor_voltage * motor_current * 0.88) / 1000.0 + noise["power"]
        vibration = (
            1.1
            + 0.0006 * pump_1_speed
            + 0.040 * motor_current
            + 0.05 * pump_pressure
            + noise["vibration"]
        )

        # 6. Temperature Sensors
        pump_temperature = (
            44.0 + 0.014 * pump_1_speed + 0.15 * pump_pressure - 0.10 * pump_flow + noise["pump_temperature"]
        )
        motor_temperature = (
            40.0 + 1.20 * motor_current + 0.005 * pump_1_speed + noise["motor_temperature"]
        )
        temperature = (
            21.5
            + 0.07 * pump_temperature
            + 0.03 * (recycle_flow - 8.0)
            + 2.5 * diurnal
            + noise["temperature"]
        )

        # 7. Quality Sensors
        pH = 7.1 + 0.08 * (dosing_rate - 12.0) - 0.015 * (inlet_flow - 60.0) + noise["pH"]
        conductivity = (
            520.0 + 14.0 * dosing_rate + 4.0 * recycle_flow - 1.2 * inlet_flow + noise["conductivity"]
        )
        turbidity = (
            2.4 + 0.035 * inlet_flow - 0.30 * (filter_pressure - 1.8) + noise["turbidity"]
        )
        chlorine = 1.5 + 0.070 * dosing_rate - 0.008 * inlet_flow + noise["chlorine"]

        return {
            "pump_1_speed": pump_1_speed,
            "pump_2_speed": pump_2_speed,
            "valve_1_position": valve_1_pos,
            "valve_2_position": valve_2_pos,
            "dosing_rate": dosing_rate,
            "pump_flow": pump_flow,
            "inlet_flow": inlet_flow,
            "outlet_flow": outlet_flow,
            "recycle_flow": recycle_flow,
            "pressure": pressure,
            "inlet_pressure": inlet_pressure,
            "pump_pressure": pump_pressure,
            "filter_pressure": filter_pressure,
            "tank_1_level": tank_1_level,
            "tank_2_level": tank_2_level,
            "tank_3_level": tank_3_level,
            "reservoir_level": reservoir_level,
            "temperature": temperature,
            "pump_temperature": pump_temperature,
            "motor_temperature": motor_temperature,
            "pH": pH,
            "conductivity": conductivity,
            "turbidity": turbidity,
            "chlorine": chlorine,
            "power": power,
            "motor_current": motor_current,
            "motor_voltage": motor_voltage,
            "vibration": vibration,
        }

    def _inject_anomaly(
        self,
        signals: dict[str, np.ndarray],
        anomaly_type: str,
        start_idx: int,
        duration: int,
        affected_sensors: list[str],
        severity: float,
    ) -> None:
        """Inject physical perturbations for specified anomaly type across target sensors."""
        idx_range = slice(start_idx, start_idx + duration)
        for s_name in affected_sensors:
            if s_name not in signals:
                continue
            cfg = self.sensor_configs.get(s_name, SENSOR_CONFIGS.get(s_name))
            normal_range = (cfg.normal_max - cfg.normal_min) if cfg else 1.0
            normal_std = cfg.noise_std if cfg else 0.1

            if anomaly_type == "sudden_spike":
                signals[s_name][idx_range] += normal_range * 0.35 * severity
            elif anomaly_type == "sudden_drop":
                signals[s_name][idx_range] -= normal_range * 0.35 * severity
            elif anomaly_type == "gradual_drift":
                ramp = np.linspace(0, normal_range * 0.40 * severity, duration)
                signals[s_name][idx_range] += ramp
            elif anomaly_type == "sensor_bias":
                signals[s_name][idx_range] += normal_range * 0.25 * severity
            elif anomaly_type == "increased_noise":
                noise_boost = self.rng.normal(0, normal_std * 8.0 * severity, duration)
                signals[s_name][idx_range] += noise_boost
            elif anomaly_type == "sensor_dropout":
                signals[s_name][idx_range] = np.nan
            elif anomaly_type == "stuck_sensor":
                frozen_val = signals[s_name][start_idx]
                signals[s_name][idx_range] = frozen_val
            elif anomaly_type == "oscillation":
                osc = (normal_range * 0.20 * severity) * np.sin(
                    2 * np.pi * np.arange(duration) / 4.0
                )
                signals[s_name][idx_range] += osc
            elif anomaly_type == "correlated_multi_sensor":
                signals[s_name][idx_range] -= normal_range * 0.30 * severity
            elif anomaly_type == "gradual_process_degradation":
                ramp = np.linspace(0, normal_range * 0.35 * severity, duration)
                signals[s_name][idx_range] += ramp

    def generate(
        self,
        days: int = 30,
        sampling_interval_minutes: int = 1,
        anomaly_rate: float = 0.04,
        sensors_per_type: int = 1,
        n_points: int | None = None,
        start_time: datetime | None = None,
        anomaly_count: int | None = None,
        min_anomaly_gap: int = 40,
        severity: float = 1.0,
    ) -> pd.DataFrame:
        """Generate a full synthetic Industrial IoT sensor dataset.

        Args:
            days: Number of simulated days.
            sampling_interval_minutes: Sampling resolution in minutes.
            anomaly_rate: Target fraction of total time points marked anomalous.
            sensors_per_type: Number of sensor instances per type.
            n_points: Optional total point count override.
            start_time: Optional starting timestamp. Defaults to deterministic epoch (2024-01-01).
            anomaly_count: Optional override for number of anomaly windows.
            min_anomaly_gap: Minimum timesteps between successive anomaly windows.
            severity: Magnitude scaling factor for injected anomalies.

        Returns:
            Long-format DataFrame with columns:
            ``timestamp``, ``sensor_type``, ``sensor_id``, ``value``, ``is_anomaly``.
        """
        if n_points is None:
            n_points = days * 24 * 60 // sampling_interval_minutes

        start_time = start_time or datetime(2024, 1, 1, 0, 0)
        timestamps = [
            start_time + timedelta(minutes=i * sampling_interval_minutes) for i in range(n_points)
        ]

        # 1. Base process simulation
        signals = self._simulate_process(n_points, sampling_interval_minutes)
        labels = np.zeros(n_points, dtype=int)
        self.anomaly_metadata = []

        # 2. Plan Anomaly Windows
        # Split layout: Train [0, 70%), Val [70%, 85%), Test [85%, 100%)
        # Train is kept 100% normal so unsupervised models learn uncorrupted process physics.
        # Val and Test receive controlled, non-saturating anomaly windows.
        dur_scale = max(6, min(30, int(n_points * 0.015)))

        val_windows = [
            ("sudden_spike", 0.73, dur_scale, ["pump_pressure"]),
            ("gradual_drift", 0.78, dur_scale + 5, ["pump_temperature"]),
            ("correlated_multi_sensor", 0.82, dur_scale, ["pump_1_speed", "pump_flow", "pressure", "motor_current"]),
        ]
        test_windows = [
            ("sudden_drop", 0.87, dur_scale, ["pump_flow"]),
            ("sensor_bias", 0.90, dur_scale + 2, ["pressure"]),
            ("increased_noise", 0.93, dur_scale + 4, ["vibration"]),
            ("stuck_sensor", 0.955, dur_scale, ["inlet_flow"]),
            ("gradual_process_degradation", 0.975, dur_scale + 8, ["motor_current", "motor_temperature", "vibration", "power"]),
        ]

        planned_windows = []
        for a_type, pos_frac, dur, aff_sensors in val_windows + test_windows:
            start_idx = int(pos_frac * n_points)
            dur = max(3, min(dur, n_points - start_idx - 1))
            planned_windows.append((a_type, start_idx, dur, aff_sensors))

        anomaly_id = 1
        for a_type, s_idx, dur, aff_sensors in planned_windows:
            e_idx = s_idx + dur
            labels[s_idx:e_idx] = 1

            self._inject_anomaly(signals, a_type, s_idx, dur, aff_sensors, severity)

            meta = AnomalyMetadata(
                anomaly_id=anomaly_id,
                anomaly_type=a_type,
                start_time=timestamps[s_idx].isoformat(),
                end_time=timestamps[e_idx - 1].isoformat(),
                start_idx=s_idx,
                end_idx=e_idx,
                affected_sensors=aff_sensors,
                severity=severity,
            )
            self.anomaly_metadata.append(meta)
            anomaly_id += 1

        # Format long-format DataFrame
        records: list[dict] = []
        for sensor_type, values in signals.items():
            for s_num in range(sensors_per_type):
                s_id = f"{sensor_type}_{s_num:02d}"
                for i in range(n_points):
                    records.append(
                        {
                            "timestamp": timestamps[i],
                            "sensor_type": sensor_type,
                            "sensor_id": s_id,
                            "value": float(values[i]),
                            "is_anomaly": int(labels[i]),
                        }
                    )

        logger.info(
            "Generated %d records across %d sensor types (%d anomalous points, %.2f%%)",
            len(records),
            len(signals),
            int(labels.sum()),
            float(labels.mean() * 100.0),
        )
        return pd.DataFrame(records)

    def compute_correlations(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute Pearson correlations between physically coupled sensor pairs."""
        pivoted = df.pivot_table(
            index="timestamp", columns="sensor_id", values="value", aggfunc="first"
        )
        pairs = [
            ("pump_1_speed_00", "pump_flow_00"),
            ("motor_current_00", "power_00"),
            ("motor_current_00", "motor_temperature_00"),
            ("pump_flow_00", "pump_pressure_00"),
            ("valve_1_position_00", "pump_flow_00"),
            ("dosing_rate_00", "pH_00"),
            ("dosing_rate_00", "chlorine_00"),
        ]
        corrs: dict[str, float] = {}
        for s1, s2 in pairs:
            if s1 in pivoted.columns and s2 in pivoted.columns:
                c = float(pivoted[s1].corr(pivoted[s2]))
                corrs[f"{s1} <-> {s2}"] = c
        return corrs

    def validate_dataset(
        self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> dict[str, Any]:
        """Validate dataset quality and consistency according to benchmark standards."""
        unique_sensors = df["sensor_id"].nunique()
        unique_timestamps = df["timestamp"].nunique()
        total_samples = unique_timestamps

        ts_series = pd.to_datetime(df["timestamp"].drop_duplicates())
        is_ordered = bool(ts_series.is_monotonic_increasing)
        no_duplicate_ts = (len(ts_series) == len(df) // unique_sensors)

        labels = df.groupby("timestamp")["is_anomaly"].max()
        n = len(labels)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        y_train = labels.iloc[:train_end].values
        y_val = labels.iloc[train_end:val_end].values
        y_test = labels.iloc[val_end:].values

        val_has_both = len(np.unique(y_val)) >= 2
        test_has_both = len(np.unique(y_test)) >= 2

        windows = []
        in_window = False
        w_start = 0
        for i, val in enumerate(labels.values):
            if val == 1 and not in_window:
                in_window = True
                w_start = i
            elif val == 0 and in_window:
                in_window = False
                windows.append((w_start, i - 1, i - w_start))
        if in_window:
            windows.append((w_start, len(labels) - 1, len(labels) - w_start))

        durations = [w[2] for w in windows]
        return {
            "total_samples": total_samples,
            "feature_count": unique_sensors,
            "timestamps_ordered": is_ordered,
            "no_duplicate_timestamps": no_duplicate_ts,
            "numeric_columns_valid": bool(np.issubdtype(df["value"].dtype, np.number)),
            "anomaly_samples": int(labels.sum()),
            "anomaly_percentage": float(labels.mean() * 100.0),
            "anomaly_windows_count": len(windows),
            "min_anomaly_duration": min(durations) if durations else 0,
            "max_anomaly_duration": max(durations) if durations else 0,
            "avg_anomaly_duration": float(np.mean(durations)) if durations else 0.0,
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "train_anomalies": int(y_train.sum()),
            "val_anomalies": int(y_val.sum()),
            "test_anomalies": int(y_test.sum()),
            "val_has_both_classes": val_has_both,
            "test_has_both_classes": test_has_both,
        }

    def save_to_csv(self, df: pd.DataFrame, path: str) -> None:
        """Persist the generated DataFrame to CSV.

        Args:
            df: The generated sensor DataFrame.
            path: Output file path (parent directories created automatically).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)
