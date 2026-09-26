"""Real-time anomaly scorer with sliding window buffers."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.streaming.simulator import SensorReading
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ScoringResult:
    """Result of scoring a single sensor reading.

    Attributes:
        timestamp: When the reading was taken.
        sensor_id: Source sensor.
        value: Raw sensor value.
        anomaly_score: Ensemble anomaly score in [0, 1].
        is_anomaly: Whether the score exceeds the anomaly threshold.
        method_scores: Per-model score breakdown.
        feature_contributions: Per-feature contribution breakdown.
        latency_ms: Scoring latency in milliseconds.
    """

    timestamp: datetime
    sensor_id: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    method_scores: dict[str, float] = field(default_factory=dict)
    feature_contributions: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


class RealTimeScorer:
    """Score incoming sensor readings using trained models.

    Maintains per-sensor sliding window buffers and produces ensemble
    predictions once readings from all sensors are available.

    Args:
        ensemble_detector: Trained :class:`EnsembleDetector`.
        preprocessor: Fitted :class:`DataPreprocessor`.
        window_minutes: Buffer size per sensor.
        db_path: Optional SQLite path for anomaly logging.
    """

    def __init__(
        self,
        ensemble_detector: Any,
        preprocessor: Any,
        window_minutes: int = 60,
        db_path: str | None = None,
    ) -> None:
        self.ensemble = ensemble_detector
        self.preprocessor = preprocessor
        self.window_minutes = window_minutes
        self.db_path = db_path

        self.buffers: dict[str, deque] = {}
        self.value_buffers: dict[str, deque] = {}

        self.on_anomaly: Callable[[ScoringResult], Coroutine] | None = None

    def _update_buffer(self, sensor_id: str, timestamp: datetime, value: float) -> None:
        """Append a reading to the per-sensor sliding window."""
        if sensor_id not in self.buffers:
            self.buffers[sensor_id] = deque(maxlen=self.window_minutes)
            self.value_buffers[sensor_id] = deque(maxlen=self.window_minutes)

        self.buffers[sensor_id].append(timestamp)
        self.value_buffers[sensor_id].append(value)

    def _get_feature_vector(self) -> np.ndarray | None:
        """Build a single feature vector from the latest value of each sensor.

        Returns ``None`` until readings from all expected sensors have arrived.
        """
        if not self.value_buffers:
            return None

        # Only produce a vector when we have data for every expected feature
        expected = len(self.preprocessor.feature_names)
        if len(self.value_buffers) < expected:
            return None

        features: list[float] = []
        for sensor_id in sorted(self.value_buffers.keys()):
            if self.value_buffers[sensor_id]:
                features.append(self.value_buffers[sensor_id][-1])
            else:
                return None

        return np.array(features).reshape(1, -1)

    async def score(self, reading: SensorReading) -> ScoringResult | None:
        """Score a single sensor reading.

        Args:
            reading: Incoming :class:`SensorReading`.

        Returns:
            :class:`ScoringResult` if all sensor buffers have data, else ``None``.
        """
        start_time = time.perf_counter()

        self._update_buffer(reading.sensor_id, reading.timestamp, reading.value)

        feature_vector = self._get_feature_vector()
        if feature_vector is None:
            return None

        X_scaled = self.preprocessor.transform(
            pd.DataFrame(feature_vector, columns=self.preprocessor.feature_names)
        )

        result = self.ensemble.predict(X_scaled)

        latency_ms = (time.perf_counter() - start_time) * 1000

        scoring_result = ScoringResult(
            timestamp=reading.timestamp,
            sensor_id=reading.sensor_id,
            value=reading.value,
            anomaly_score=float(result.scores[0]),
            is_anomaly=bool(result.predictions[0]),
            method_scores={k: float(v[0]) for k, v in result.method_scores.items()},
            feature_contributions={
                k: float(v[0]) for k, v in result.feature_contributions.items()
            },
            latency_ms=latency_ms,
        )

        if self.db_path and scoring_result.is_anomaly:
            self._log_anomaly(scoring_result)

        if scoring_result.is_anomaly and self.on_anomaly:
            await self.on_anomaly(scoring_result)

        return scoring_result

    def _log_anomaly(self, result: ScoringResult) -> None:
        """Persist detected anomaly to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO anomaly_log
               (timestamp, sensor_id, anomaly_score, detection_method, contributing_features)
               VALUES (?, ?, ?, ?, ?)""",
            (
                result.timestamp.isoformat(),
                result.sensor_id,
                result.anomaly_score,
                "ensemble",
                json.dumps(result.feature_contributions),
            ),
        )
        conn.commit()
        conn.close()
