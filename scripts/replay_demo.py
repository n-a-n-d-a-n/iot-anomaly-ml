"""CLI script to run and benchmark the Streaming Replay engine on the Industrial IoT dataset.

Usage:
    python -m scripts.replay_demo [--max-samples SAMPLES] [--speed SPEED]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import joblib
import torch

from src.analysis.root_cause import RootCauseAnalyzer
from src.data.preprocessor import DataPreprocessor
from src.models.autoencoder import AutoencoderDetector
from src.models.dbscan_detector import DBSCANDetector
from src.models.ensemble import EnsembleDetector
from src.models.gmm_detector import GMMDetector
from src.models.isolation_forest import IsolationForestDetector
from src.streaming.replay import StreamingReplay

DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
DATA_CSV = DATA_DIR / "generated" / "sensor_data.csv"


def load_pipeline():
    """Load trained models and fitted preprocessor."""
    iso = IsolationForestDetector.load(str(MODELS_DIR / "isolation_forest.joblib"))
    ae = AutoencoderDetector.load(str(MODELS_DIR / "autoencoder.pt"))
    dbscan = DBSCANDetector.load(str(MODELS_DIR / "dbscan.joblib"))
    gmm = GMMDetector.load(str(MODELS_DIR / "gmm.joblib"))

    ensemble = EnsembleDetector()
    ensemble.add_model("isolation_forest", iso)
    ensemble.add_model("autoencoder", ae)
    ensemble.add_model("dbscan", dbscan)
    ensemble.add_model("gmm", gmm)

    preprocessor = DataPreprocessor(scaler_type="standard")
    return ensemble, preprocessor


async def run_replay(max_samples: int = 500, speed: float = 0.0):
    print("=" * 60)
    print("INDUSTRIAL IoT STREAMING REPLAY & BENCHMARK DEMO")
    print("=" * 60)

    ensemble, preprocessor = load_pipeline()

    # Pre-fit preprocessor on training data to mimic production environment
    import pandas as pd
    df_raw = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    pivoted = preprocessor.pivot_sensors(df_raw)
    preprocessor.fit_transform(pivoted.iloc[:7056])  # Fit on training split only
    ensemble.calibrate_threshold(
        preprocessor.transform(pivoted.iloc[7056:8568]),
        pivoted["is_anomaly"].iloc[7056:8568].values,
    )

    analyzer = RootCauseAnalyzer(sensor_names=preprocessor.feature_names)
    analyzer.compute_reference_stats(pivoted.iloc[:7056])
    analyzer.compute_correlation_matrix(pivoted)

    replay = StreamingReplay(
        data_path=DATA_CSV,
        ensemble=ensemble,
        preprocessor=preprocessor,
        analyzer=analyzer,
        speed_multiplier=speed,
    )

    print(f"Dataset: {DATA_CSV}")
    print(f"Features: {len(preprocessor.feature_names)} sensors")
    print(f"Calibrated threshold: {ensemble.threshold:.4f}")
    print(f"Replaying {max_samples} samples (speed={speed}x)...\n")

    anomaly_count = 0
    async for event in replay.stream_replay(max_samples=max_samples):
        if event.is_anomaly:
            anomaly_count += 1
            expl = event.explanation
            top_s = expl.top_contributing_sensors[0].sensor_id if expl and expl.top_contributing_sensors else "unknown"
            pattern = expl.anomaly_type if expl else "unknown"
            print(
                f"[ALERT #{anomaly_count}] {event.timestamp.isoformat()} | Score: {event.ensemble_score:.3f} "
                f"> Thresh: {event.threshold:.3f} | Top: {top_s} | Pattern: {pattern} | Latency: {event.latency_ms:.2f} ms"
            )

    m = replay.metrics
    print("\n" + "=" * 60)
    print("STREAMING REPLAY PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total samples processed:   {m.total_samples}")
    print(f"Anomalies detected:        {m.total_anomalies}")
    print(f"Total processing time:     {m.total_processing_time_s:.4f} s")
    print(f"Average latency:           {m.avg_latency_ms:.2f} ms/sample")
    print(f"Peak (max) latency:        {m.max_latency_ms:.2f} ms")
    print(f"Minimum latency:           {m.min_latency_ms:.2f} ms")
    print(f"Throughput:                {m.throughput_samples_per_sec:.1f} samples/sec")
    print(f"Target (<100 ms/sample):   {'PASSED' if m.avg_latency_ms < 100 else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming replay demo")
    parser.add_argument("--max-samples", type=int, default=500, help="Number of samples to replay")
    parser.add_argument("--speed", type=float, default=0.0, help="Speed multiplier (0 = max compute speed)")
    args = parser.parse_args()
    asyncio.run(run_replay(max_samples=args.max_samples, speed=args.speed))
