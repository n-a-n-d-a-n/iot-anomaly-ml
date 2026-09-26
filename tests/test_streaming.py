"""Tests for the async streaming pipeline and real-time scorer."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.data.generator import IoTDataGenerator
from src.streaming.scorer import RealTimeScorer, ScoringResult
from src.streaming.simulator import SensorReading, StreamSimulator

# ---------------------------------------------------------------------------
# Stream simulator tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_df() -> pd.DataFrame:
    gen = IoTDataGenerator(seed=0)
    return gen.generate(days=1, sensors_per_type=1).head(40)


class TestStreamSimulator:
    @pytest.mark.asyncio
    async def test_yields_readings_in_order(self, small_df: pd.DataFrame) -> None:
        sim = StreamSimulator(small_df, speed_multiplier=1e6)
        timestamps = []
        async for reading in sim.stream():
            timestamps.append(reading.timestamp)
            if len(timestamps) >= 20:
                sim.stop()
                break
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    @pytest.mark.asyncio
    async def test_reading_has_correct_fields(self, small_df: pd.DataFrame) -> None:
        sim = StreamSimulator(small_df, speed_multiplier=1e6)
        async for reading in sim.stream():
            assert isinstance(reading, SensorReading)
            assert isinstance(reading.timestamp, datetime)
            assert isinstance(reading.value, float)
            sim.stop()
            break

    @pytest.mark.asyncio
    async def test_stop_halts_stream(self, small_df: pd.DataFrame) -> None:
        sim = StreamSimulator(small_df, speed_multiplier=1e6)
        count = 0
        async for _ in sim.stream():
            count += 1
            if count >= 5:
                sim.stop()
        assert count >= 5


# ---------------------------------------------------------------------------
# Fake models for scorer tests
# ---------------------------------------------------------------------------


class _FakeEnsemble:
    def __init__(self) -> None:
        self.threshold = 0.5

    def predict(self, X: np.ndarray):
        from src.models.ensemble import EnsembleResult

        n = len(X)
        return EnsembleResult(
            scores=np.full(n, 0.3),
            predictions=np.zeros(n, dtype=int),
            method_scores={"fake": np.full(n, 0.3)},
            feature_contributions={"f0": np.full(n, 0.1)},
        )


class _FakeEnsembleAnomaly:
    def __init__(self) -> None:
        self.threshold = 0.5

    def predict(self, X: np.ndarray):
        from src.models.ensemble import EnsembleResult

        n = len(X)
        return EnsembleResult(
            scores=np.full(n, 0.9),
            predictions=np.ones(n, dtype=int),
            method_scores={"fake": np.full(n, 0.9)},
            feature_contributions={"f0": np.full(n, 0.8)},
        )


class _FakePreprocessor:
    def __init__(self) -> None:
        self.feature_names = ["sensor_a", "sensor_b"]
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df.values


# ---------------------------------------------------------------------------
# Real-time scorer tests
# ---------------------------------------------------------------------------


class TestRealTimeScorer:
    def _make_reading(self, sensor_id: str = "sensor_a", value: float = 70.0) -> SensorReading:
        return SensorReading(
            timestamp=datetime.now(),
            sensor_id=sensor_id,
            sensor_type="temperature",
            value=value,
        )

    @pytest.mark.asyncio
    async def test_returns_none_until_all_sensors(self) -> None:
        scorer = RealTimeScorer(_FakeEnsemble(), _FakePreprocessor())
        result = await scorer.score(self._make_reading("sensor_a"))
        # Only 1 of 2 expected sensors present
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_result_when_all_sensors(self) -> None:
        scorer = RealTimeScorer(_FakeEnsemble(), _FakePreprocessor())
        await scorer.score(self._make_reading("sensor_a"))
        result = await scorer.score(self._make_reading("sensor_b"))
        assert result is not None
        assert isinstance(result, ScoringResult)

    @pytest.mark.asyncio
    async def test_scoring_result_fields(self) -> None:
        scorer = RealTimeScorer(_FakeEnsemble(), _FakePreprocessor())
        await scorer.score(self._make_reading("sensor_a"))
        result = await scorer.score(self._make_reading("sensor_b"))
        assert result.anomaly_score == pytest.approx(0.3)
        assert result.is_anomaly is False
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_buffer_update(self) -> None:
        scorer = RealTimeScorer(_FakeEnsemble(), _FakePreprocessor())
        await scorer.score(self._make_reading("sensor_a", 10.0))
        assert len(scorer.value_buffers["sensor_a"]) == 1
        assert scorer.value_buffers["sensor_a"][-1] == 10.0

    @pytest.mark.asyncio
    async def test_anomaly_callback_triggered(self) -> None:
        callback = AsyncMock()
        scorer = RealTimeScorer(_FakeEnsembleAnomaly(), _FakePreprocessor())
        scorer.on_anomaly = callback
        await scorer.score(self._make_reading("sensor_a"))
        await scorer.score(self._make_reading("sensor_b"))
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_anomaly_logged_to_db(self, tmp_path) -> None:
        from src.utils.database import init_db

        db_path = str(tmp_path / "test.db")
        init_db(db_path)

        scorer = RealTimeScorer(_FakeEnsembleAnomaly(), _FakePreprocessor(), db_path=db_path)
        await scorer.score(self._make_reading("sensor_a"))
        await scorer.score(self._make_reading("sensor_b"))

        import sqlite3

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM anomaly_log").fetchone()[0]
        conn.close()
        assert count >= 1

    @pytest.mark.asyncio
    async def test_latency_reasonable(self) -> None:
        scorer = RealTimeScorer(_FakeEnsemble(), _FakePreprocessor())
        await scorer.score(self._make_reading("sensor_a"))
        result = await scorer.score(self._make_reading("sensor_b"))
        assert result.latency_ms < 100
