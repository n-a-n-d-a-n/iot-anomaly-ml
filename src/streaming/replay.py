"""Streaming replay module for chronologically simulating IoT sensor streams.

Replays generated industrial sensor datasets in strict chronological sequence,
feeding reading vectors through preprocessing and ensemble models, tracking
real-time scoring latency, detecting anomalies, and emitting rich explanations.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.root_cause import AnomalyExplanation, RootCauseAnalyzer
from src.data.preprocessor import DataPreprocessor
from src.models.ensemble import EnsembleDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ReplayMetrics:
    """Latency and throughput performance metrics measured during streaming replay."""

    total_samples: int = 0
    total_anomalies: int = 0
    total_processing_time_s: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """Average inference and explainability latency per time-step in milliseconds."""
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def max_latency_ms(self) -> float:
        """Peak latency per sample in milliseconds."""
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def min_latency_ms(self) -> float:
        """Minimum latency per sample in milliseconds."""
        return float(np.min(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def throughput_samples_per_sec(self) -> float:
        """Effective processing throughput."""
        return (
            (self.total_samples / self.total_processing_time_s)
            if self.total_processing_time_s > 0
            else 0.0
        )


@dataclass
class ReplayEvent:
    """Event emitted at each time step during streaming replay."""

    timestamp: datetime
    sample_index: int
    ensemble_score: float
    threshold: float
    is_anomaly: bool
    model_scores: dict[str, float]
    feature_contributions: dict[str, float]
    raw_values: dict[str, float]
    latency_ms: float
    explanation: AnomalyExplanation | None = None


class StreamingReplay:
    """Chronological event-driven stream replay engine for Industrial IoT evaluation.

    Args:
        data_path: Path to the generated long or wide format CSV file.
        ensemble: Pre-trained :class:`EnsembleDetector` instance.
        preprocessor: Fitted :class:`DataPreprocessor` instance.
        analyzer: Optional pre-configured :class:`RootCauseAnalyzer`.
        speed_multiplier: Playback acceleration factor (e.g. 10.0 = 10x real-time; 0 = max compute speed).
    """

    def __init__(
        self,
        data_path: str | Path,
        ensemble: EnsembleDetector,
        preprocessor: DataPreprocessor,
        analyzer: RootCauseAnalyzer | None = None,
        speed_multiplier: float = 0.0,
    ) -> None:
        self.data_path = Path(data_path)
        self.ensemble = ensemble
        self.preprocessor = preprocessor
        self.speed_multiplier = speed_multiplier
        self._running: bool = False
        self.metrics = ReplayMetrics()

        # Load and pivot dataset to chronological wide format
        df_raw = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        if "sensor_id" in df_raw.columns:
            self.pivoted_df = self.preprocessor.pivot_sensors(df_raw)
        else:
            self.pivoted_df = df_raw.sort_values("timestamp")

        # Initialize analyzer with normal baseline statistics if needed
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            feature_names = [
                c for c in self.pivoted_df.columns if c not in ("timestamp", "is_anomaly")
            ]
            self.analyzer = RootCauseAnalyzer(sensor_names=feature_names)
            self.analyzer.compute_reference_stats(self.pivoted_df)
            self.analyzer.compute_correlation_matrix(self.pivoted_df)

    async def stream_replay(
        self,
        max_samples: int | None = None,
        on_event: Callable[[ReplayEvent], Any] | None = None,
    ) -> AsyncIterator[ReplayEvent]:
        """Stream sensor observations chronologically through the ML scoring pipeline.

        Args:
            max_samples: Optional limit on the number of samples to replay.
            on_event: Optional callback fired on each processed event.

        Yields:
            Processed :class:`ReplayEvent` instances in temporal order.
        """
        self._running = True
        self.metrics = ReplayMetrics()
        feature_cols = [c for c in self.pivoted_df.columns if c not in ("timestamp", "is_anomaly")]

        rows_to_process = self.pivoted_df.head(max_samples) if max_samples else self.pivoted_df
        prev_time: pd.Timestamp | None = None

        t_sim_start = time.perf_counter()

        for idx, row in rows_to_process.iterrows():
            if not self._running:
                break

            current_time = pd.to_datetime(row["timestamp"])

            # Pace replay according to speed multiplier
            if self.speed_multiplier > 0 and prev_time is not None:
                sim_dt = (current_time - prev_time).total_seconds()
                delay = sim_dt / self.speed_multiplier
                if delay > 0.001:
                    await asyncio.sleep(min(delay, 0.5))

            prev_time = current_time

            # Benchmark inference latency
            t0 = time.perf_counter()

            raw_dict = {col: float(row[col]) for col in feature_cols}
            single_sample_df = pd.DataFrame([raw_dict])

            # Scale and predict
            X_scaled = self.preprocessor.transform(single_sample_df)
            result = self.ensemble.predict(X_scaled, compute_contributions=False)

            score = float(result.scores[0])
            is_anom = bool(result.predictions[0])
            method_scores = {k: float(v[0]) for k, v in result.method_scores.items()}

            # Generate explanation and contributions only if anomalous
            explanation = None
            contributions: dict[str, float] = {}
            if is_anom:
                contrib_res = self.ensemble.predict(X_scaled, compute_contributions=True)
                contributions = {k: float(v[0]) for k, v in contrib_res.feature_contributions.items()}
                explanation = self.analyzer.explain_anomaly(
                    timestamp=current_time,
                    anomaly_score=score,
                    feature_contributions=contributions,
                    current_values=raw_dict,
                    model_scores=method_scores,
                    threshold=self.ensemble.threshold,
                )

            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Record metrics
            self.metrics.total_samples += 1
            if is_anom:
                self.metrics.total_anomalies += 1
            self.metrics.latencies_ms.append(t_elapsed_ms)

            event = ReplayEvent(
                timestamp=current_time.to_pydatetime(),
                sample_index=int(idx),
                ensemble_score=score,
                threshold=self.ensemble.threshold,
                is_anomaly=is_anom,
                model_scores=method_scores,
                feature_contributions=contributions,
                raw_values=raw_dict,
                latency_ms=t_elapsed_ms,
                explanation=explanation,
            )

            if on_event:
                if asyncio.iscoroutinefunction(on_event):
                    await on_event(event)
                else:
                    on_event(event)

            yield event

        self.metrics.total_processing_time_s = time.perf_counter() - t_sim_start
        logger.info(
            "Replay complete: %d samples, %d anomalies, avg latency: %.2f ms, max latency: %.2f ms",
            self.metrics.total_samples,
            self.metrics.total_anomalies,
            self.metrics.avg_latency_ms,
            self.metrics.max_latency_ms,
        )

    def stop(self) -> None:
        """Halt the active replay loop."""
        self._running = False
