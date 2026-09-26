"""FastAPI REST endpoints for the anomaly detection service."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

config = Config.load()
DB_PATH = config.get("database.path", "data/anomaly_detection.db")

# Global state populated on startup
models: dict[str, Any] = {}
preprocessor: Any = None
ensemble: Any = None


# ----- Lifecycle events -----


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load trained models on startup."""
    global models, preprocessor, ensemble  # noqa: PLW0603

    model_dir = Path("data/models")
    if not model_dir.exists():
        logger.info("No models directory found; endpoints will return 503 until trained")
        yield
        return

    try:
        import joblib

        from src.models.autoencoder import AutoencoderDetector
        from src.models.dbscan_detector import DBSCANDetector
        from src.models.ensemble import EnsembleDetector
        from src.models.gmm_detector import GMMDetector
        from src.models.isolation_forest import IsolationForestDetector

        if (model_dir / "isolation_forest.joblib").exists():
            models["isolation_forest"] = IsolationForestDetector.load(
                str(model_dir / "isolation_forest.joblib")
            )
        if (model_dir / "autoencoder.pt").exists():
            models["autoencoder"] = AutoencoderDetector.load(str(model_dir / "autoencoder.pt"))
        if (model_dir / "dbscan.joblib").exists():
            models["dbscan"] = DBSCANDetector.load(str(model_dir / "dbscan.joblib"))
        if (model_dir / "gmm.joblib").exists():
            models["gmm"] = GMMDetector.load(str(model_dir / "gmm.joblib"))
        if (model_dir / "preprocessor.joblib").exists():
            preprocessor = joblib.load(str(model_dir / "preprocessor.joblib"))

        if models:
            ensemble = EnsembleDetector()
            for name, model in models.items():
                ensemble.add_model(name, model)
            logger.info("Loaded %d models into ensemble", len(models))
    except Exception:
        logger.exception("Error loading models")

    yield


app = FastAPI(
    title="IoT Anomaly Detection API",
    description="REST API for real-time anomaly detection on IoT sensor data",
    version="1.0.0",
    lifespan=lifespan,
)


# ----- Pydantic schemas -----


class SensorReadingIn(BaseModel):
    """Incoming sensor reading payload."""

    timestamp: datetime
    sensor_id: str
    sensor_type: str
    value: float


class SensorReadingBatch(BaseModel):
    """Batch of sensor readings."""

    readings: list[SensorReadingIn]


class AnomalyResponse(BaseModel):
    """Anomaly scoring result."""

    timestamp: datetime
    sensor_id: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    method_scores: dict[str, float]
    feature_contributions: dict[str, float]


class AlertResponse(BaseModel):
    """Alert record from the database."""

    id: int
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    resolved: bool


class ModelStatusResponse(BaseModel):
    """Status of a single model."""

    model_name: str
    status: str
    last_trained: datetime | None = None
    metrics: dict[str, float] | None = None


# ----- Endpoints -----


@app.get("/health")
async def health_check() -> dict:
    """Liveness / readiness probe."""
    return {"status": "healthy", "models_loaded": len(models)}


@app.post("/score", response_model=AnomalyResponse)
async def score_reading(reading: SensorReadingIn) -> AnomalyResponse:
    """Score a single sensor reading for anomalies."""
    if ensemble is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    import pandas as pd

    df = pd.DataFrame([{reading.sensor_id: reading.value}])
    try:
        X = preprocessor.transform(df[[reading.sensor_id]])
        result = ensemble.predict(X)
        return AnomalyResponse(
            timestamp=reading.timestamp,
            sensor_id=reading.sensor_id,
            value=reading.value,
            anomaly_score=float(result.scores[0]),
            is_anomaly=bool(result.predictions[0]),
            method_scores={k: float(v[0]) for k, v in result.method_scores.items()},
            feature_contributions={
                k: float(v[0]) for k, v in result.feature_contributions.items()
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/score/batch", response_model=list[AnomalyResponse])
async def score_batch(batch: SensorReadingBatch) -> list[AnomalyResponse]:
    """Score a batch of sensor readings."""
    results: list[AnomalyResponse] = []
    for reading in batch.readings:
        result = await score_reading(reading)
        results.append(result)
    return results


@app.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    limit: int = 100,
    severity: str | None = None,
    resolved: bool | None = None,
) -> list[AlertResponse]:
    """Retrieve alert history with optional filters."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM alert_history WHERE 1=1"
    params: list = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if resolved is not None:
        query += " AND resolved = ?"
        params.append(int(resolved))

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        AlertResponse(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            alert_type=row["alert_type"],
            severity=row["severity"],
            message=row["message"],
            resolved=bool(row["resolved"]),
        )
        for row in rows
    ]


@app.put("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int) -> dict:
    """Mark an alert as resolved."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE alert_history SET resolved = 1, resolution_timestamp = ? WHERE id = ?",
        (datetime.now().isoformat(), alert_id),
    )
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Alert not found")
    conn.commit()
    conn.close()
    return {"status": "resolved", "alert_id": alert_id}


@app.get("/models/status", response_model=list[ModelStatusResponse])
async def get_model_status() -> list[ModelStatusResponse]:
    """Return the load status of each model."""
    statuses: list[ModelStatusResponse] = []
    for name in ["isolation_forest", "autoencoder", "dbscan", "gmm", "ensemble"]:
        status = "loaded" if name in models or (name == "ensemble" and ensemble) else "not_loaded"
        statuses.append(ModelStatusResponse(model_name=name, status=status))
    return statuses


@app.get("/sensors/health")
async def get_sensor_health() -> list[dict]:
    """Health status summary for all sensors."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT sensor_id,
                  COUNT(*) as total,
                  SUM(CASE WHEN is_anomaly_ground_truth = 1 THEN 1 ELSE 0 END) as anomalies
           FROM sensor_readings
           GROUP BY sensor_id"""
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "sensor_id": row[0],
            "total_readings": row[1],
            "anomaly_count": row[2],
            "anomaly_rate": row[2] / row[1] if row[1] > 0 else 0,
            "status": (
                "healthy"
                if row[2] / row[1] < 0.05
                else "warning"
                if row[2] / row[1] < 0.1
                else "critical"
            ),
        }
        for row in rows
    ]


@app.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Retrieve model evaluation metrics from data/evaluation_results.json."""
    import json
    metrics_path = Path("data/evaluation_results.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/sensors")
async def get_sensors() -> dict[str, Any]:
    """Retrieve the inventory of monitored industrial sensors and their groupings."""
    cfg = Config.load()
    sensor_confs = cfg.sensors
    groups: dict[str, list[dict[str, Any]]] = {}
    for s_id, s_conf in sensor_confs.items():
        grp = getattr(s_conf, "sensor_type", "general")
        if grp not in groups:
            groups[grp] = []
        groups[grp].append({
            "sensor_id": s_id,
            "sensor_type": grp,
            "base_mean": getattr(s_conf, "base_mean", None),
            "base_std": getattr(s_conf, "base_std", None),
            "unit": getattr(s_conf, "unit", ""),
        })
    return {
        "total_sensors": len(sensor_confs),
        "groups": groups,
        "sensors": list(sensor_confs.keys()),
    }


@app.get("/anomalies")
async def get_anomalies(limit: int = 50, resolved: bool | None = None) -> list[dict[str, Any]]:
    """Retrieve detected anomalies from the database."""
    import json
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    query = "SELECT * FROM anomaly_log WHERE 1=1"
    params: list = []
    if resolved is not None:
        query += " AND resolved = ?"
        params.append(int(resolved))
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "sensor_type": r["sensor_type"],
            "sensor_id": r["sensor_id"],
            "anomaly_score": r["anomaly_score"],
            "detection_method": r["detection_method"],
            "contributing_features": json.loads(r["contributing_features"]) if r["contributing_features"] else {},
            "resolved": bool(r["resolved"]),
        })
    return results


@app.get("/explanation/{timestamp}")
@app.get("/anomalies/{timestamp}")
async def get_anomaly_explanation(timestamp: str) -> dict[str, Any]:
    """Retrieve explainable diagnosis for an anomaly at a specific timestamp."""
    import json
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM anomaly_log WHERE timestamp = ? LIMIT 1", (timestamp,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "timestamp": row["timestamp"],
            "anomaly_score": row["anomaly_score"],
            "threshold": ensemble.threshold if ensemble else 0.6061,
            "detected": True,
            "detection_method": row["detection_method"],
            "contributing_features": json.loads(row["contributing_features"]) if row["contributing_features"] else {},
        }
    raise HTTPException(status_code=404, detail=f"No anomaly record found for timestamp: {timestamp}")
