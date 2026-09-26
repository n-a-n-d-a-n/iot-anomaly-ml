"""Tests for streaming replay engine, latency metrics, and real-time alert generation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.analysis.root_cause import RootCauseAnalyzer
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble import EnsembleDetector, EnsembleResult
from src.streaming.replay import ReplayEvent, ReplayMetrics, StreamingReplay


class _StubDetector:
    """Minimal detector stub for deterministic streaming tests."""

    def __init__(self, features: list[str]) -> None:
        self.features = features

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if np.any(np.abs(X) > 1.8):
            return np.array([0.85] * len(X))
        return np.array([0.2] * len(X))

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {f: np.array([0.5] * len(X)) for f in self.features}


@pytest.fixture()
def replay_fixture(tmp_path) -> tuple[StreamingReplay, Path]:
    # Generate 20 samples of 3 sensors
    base = datetime(2024, 1, 1, 0, 0, 0)
    features = ["sensor_a", "sensor_b", "sensor_c"]
    rows = []
    for i in range(20):
        rows.append({
            "timestamp": base + timedelta(minutes=i),
            "sensor_a": 10.0 + i,
            "sensor_b": 20.0 + (50.0 if i == 10 else 0.0),  # anomaly at index 10
            "sensor_c": 30.0,
            "is_anomaly": 1 if i == 10 else 0,
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "sensor_data.csv"
    df.to_csv(csv_path, index=False)

    preprocessor = DataPreprocessor(scaler_type="standard")
    preprocessor.fit_transform(df)

    ensemble = EnsembleDetector(weights={"stub": 1.0})
    stub = _StubDetector(features)
    ensemble.add_model("stub", stub)
    ensemble.threshold = 0.60

    analyzer = RootCauseAnalyzer(sensor_names=features)
    analyzer.compute_reference_stats(df)

    replay = StreamingReplay(
        data_path=csv_path,
        ensemble=ensemble,
        preprocessor=preprocessor,
        analyzer=analyzer,
        speed_multiplier=0.0,
    )
    return replay, csv_path


class TestStreamingReplay:
    def test_chronological_streaming(self, replay_fixture) -> None:
        replay, _ = replay_fixture

        events: list[ReplayEvent] = []
        async def run():
            async for ev in replay.stream_replay(max_samples=15):
                events.append(ev)

        asyncio.run(run())

        assert len(events) == 15
        # Verify chronological ordering
        for i in range(len(events) - 1):
            assert events[i].timestamp < events[i + 1].timestamp

    def test_anomaly_detection_and_explanation(self, replay_fixture) -> None:
        replay, _ = replay_fixture

        events: list[ReplayEvent] = []
        async def run():
            async for ev in replay.stream_replay(max_samples=20):
                events.append(ev)

        asyncio.run(run())

        anomalies = [ev for ev in events if ev.is_anomaly]
        assert len(anomalies) >= 1
        anom_ev = anomalies[0]
        assert anom_ev.ensemble_score >= replay.ensemble.threshold
        assert anom_ev.explanation is not None
        assert anom_ev.explanation.detected is True

    def test_latency_metrics_recording(self, replay_fixture) -> None:
        replay, _ = replay_fixture

        async def run():
            async for _ in replay.stream_replay(max_samples=10):
                pass

        asyncio.run(run())

        m = replay.metrics
        assert m.total_samples == 10
        assert len(m.latencies_ms) == 10
        assert m.avg_latency_ms > 0
        assert m.max_latency_ms >= m.avg_latency_ms
        assert m.throughput_samples_per_sec > 0

    def test_on_event_callback(self, replay_fixture) -> None:
        replay, _ = replay_fixture
        callback_log = []

        def on_event(ev: ReplayEvent):
            callback_log.append(ev.sample_index)

        async def run():
            async for _ in replay.stream_replay(max_samples=5, on_event=on_event):
                pass

        asyncio.run(run())
        assert len(callback_log) == 5
        assert callback_log == [0, 1, 2, 3, 4]
