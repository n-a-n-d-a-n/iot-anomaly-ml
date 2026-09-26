"""Populate the dashboard database with sensor readings and anomaly scores.

Loads trained models, runs the ensemble scorer over the generated sensor data,
and writes both raw readings and detected anomalies to SQLite.

Usage:
    python -m scripts.populate_db [--data-path DATA_CSV]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessor import DataPreprocessor
from src.models.autoencoder import AutoencoderDetector
from src.models.dbscan_detector import DBSCANDetector
from src.models.ensemble import EnsembleDetector
from src.models.gmm_detector import GMMDetector
from src.models.isolation_forest import IsolationForestDetector
from src.utils.database import init_db
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = Path("data")
GENERATED_CSV = DATA_DIR / "generated" / "sensor_data.csv"
DB_PATH = DATA_DIR / "anomaly_detection.db"
MODELS_DIR = DATA_DIR / "models"


def main(data_path: Path = GENERATED_CSV) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"No data at {data_path}. Run 'make generate-data' first.")

    # Check trained models exist
    iso_path = MODELS_DIR / "isolation_forest.joblib"
    ae_path = MODELS_DIR / "autoencoder.pt"
    dbscan_path = MODELS_DIR / "dbscan.joblib"
    gmm_path = MODELS_DIR / "gmm.joblib"
    for p in (iso_path, ae_path, dbscan_path, gmm_path):
        if not p.exists():
            raise FileNotFoundError(f"No trained model at {p}. Run 'make train' first.")

    # Load data
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    # Initialise DB with correct schema
    logger.info("Initialising database at %s", DB_PATH)
    conn = init_db(str(DB_PATH))

    # Drop old tables and recreate with correct schema
    conn.execute("DROP TABLE IF EXISTS sensor_readings")
    conn.execute("DROP TABLE IF EXISTS anomaly_log")
    conn.execute("DROP TABLE IF EXISTS alert_history")
    conn.commit()
    conn.close()
    conn = init_db(str(DB_PATH))

    # Insert sensor readings
    logger.info("Inserting %d sensor readings...", len(df))
    readings = [
        (
            row["timestamp"].isoformat(),
            row["sensor_type"],
            row["sensor_id"],
            float(row["value"]),
            int(row["is_anomaly"]),
        )
        for _, row in df.iterrows()
    ]
    conn.executemany(
        "INSERT INTO sensor_readings (timestamp, sensor_type, sensor_id, value, is_anomaly_ground_truth) "
        "VALUES (?, ?, ?, ?, ?)",
        readings,
    )
    conn.commit()
    logger.info("Inserted sensor readings")

    # Load trained models
    logger.info("Loading trained models...")
    iso = IsolationForestDetector.load(str(iso_path))
    ae = AutoencoderDetector.load(str(ae_path))
    dbscan = DBSCANDetector.load(str(dbscan_path))
    gmm = GMMDetector.load(str(gmm_path))

    ensemble = EnsembleDetector()
    ensemble.add_model("isolation_forest", iso)
    ensemble.add_model("autoencoder", ae)
    ensemble.add_model("dbscan", dbscan)
    ensemble.add_model("gmm", gmm)

    # Preprocess for scoring
    preprocessor = DataPreprocessor(scaler_type="standard")
    pivoted = preprocessor.pivot_sensors(df)
    X_scaled, y = preprocessor.fit_transform(pivoted)

    # Calibrate ensemble threshold on the data
    if y is not None:
        ensemble.calibrate_threshold(X_scaled, y)

    # Score all data with the ensemble
    logger.info("Scoring %d samples with ensemble...", len(X_scaled))
    result = ensemble.predict(X_scaled)

    # Insert anomalies into anomaly_log
    anomaly_indices = np.where(result.predictions == 1)[0]
    logger.info("Detected %d anomalies, inserting into anomaly_log...", len(anomaly_indices))

    timestamps = pivoted["timestamp"].values
    feature_names = preprocessor.feature_names

    anomaly_rows = []
    for idx in anomaly_indices:
        ts = pd.Timestamp(timestamps[idx]).isoformat()
        score = float(result.scores[idx])
        contributions = {
            feat: float(result.feature_contributions[feat][idx])
            for feat in feature_names
            if feat in result.feature_contributions
        }
        # Find which sensor contributed most
        top_sensor = (
            max(contributions, key=contributions.get) if contributions else feature_names[0]
        )
        anomaly_rows.append(
            (
                ts,
                top_sensor.split("_")[0] if "_" in top_sensor else top_sensor,
                top_sensor,
                score,
                "ensemble",
                json.dumps(contributions),
            )
        )

    conn.executemany(
        "INSERT INTO anomaly_log (timestamp, sensor_type, sensor_id, anomaly_score, detection_method, contributing_features) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        anomaly_rows,
    )
    conn.commit()

    # Insert some alerts for detected anomalies
    logger.info("Generating alerts for high-score anomalies...")
    alert_rows = []
    anomaly_id_start = 1  # autoincrement starts at 1
    for i, idx in enumerate(anomaly_indices):
        score = float(result.scores[idx])
        if score < 0.7:
            continue
        ts = pd.Timestamp(timestamps[idx]).isoformat()
        severity = "critical" if score > 0.9 else "high" if score > 0.8 else "medium"
        contributions = {
            feat: float(result.feature_contributions[feat][idx])
            for feat in feature_names
            if feat in result.feature_contributions
        }
        top_feat = max(contributions, key=contributions.get) if contributions else "unknown"
        message = f"Anomaly score {score:.2f} detected — top contributor: {top_feat}"
        alert_rows.append(
            (
                ts,
                "anomaly_detected",
                severity,
                message,
                anomaly_id_start + i,
            )
        )

    if alert_rows:
        conn.executemany(
            "INSERT INTO alert_history (timestamp, alert_type, severity, message, anomaly_id) "
            "VALUES (?, ?, ?, ?, ?)",
            alert_rows,
        )
        conn.commit()
        logger.info("Inserted %d alerts", len(alert_rows))

    conn.close()
    logger.info("Database populated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate dashboard database")
    parser.add_argument("--data-path", type=Path, default=GENERATED_CSV, help="Path to sensor CSV")
    args = parser.parse_args()
    main(data_path=args.data_path)
