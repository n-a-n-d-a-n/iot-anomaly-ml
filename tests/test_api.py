"""Tests for FastAPI REST endpoints."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.routes import app
from src.utils.config import Config
from src.utils.database import init_db


@pytest.fixture(autouse=True)
def _reset_config():
    Config.reset()
    yield
    Config.reset()


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    conn = init_db(path)
    # Seed some data
    conn.execute(
        "INSERT INTO sensor_readings (timestamp, sensor_type, sensor_id, value, is_anomaly_ground_truth) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), "temperature", "temp_00", 72.0, 0),
    )
    conn.execute(
        "INSERT INTO alert_history (timestamp, alert_type, severity, message, resolved) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), "threshold", "high", "test alert", 0),
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def client(db_path: str, monkeypatch):
    """Create a test client with patched DB path."""
    import src.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "DB_PATH", db_path)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data


class TestScoreEndpoint:
    def test_score_returns_503_without_models(self, client: TestClient) -> None:
        import src.api.routes as routes_mod

        original_ensemble = routes_mod.ensemble
        routes_mod.ensemble = None
        try:
            resp = client.post(
                "/score",
                json={
                    "timestamp": datetime.now().isoformat(),
                    "sensor_id": "temp_00",
                    "sensor_type": "temperature",
                    "value": 75.0,
                },
            )
            assert resp.status_code == 503
        finally:
            routes_mod.ensemble = original_ensemble


class TestAlertsEndpoint:
    def test_get_alerts(self, client: TestClient) -> None:
        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_get_alerts_with_severity_filter(self, client: TestClient) -> None:
        resp = client.get("/alerts?severity=high")
        assert resp.status_code == 200
        data = resp.json()
        for alert in data:
            assert alert["severity"] == "high"

    def test_get_alerts_with_resolved_filter(self, client: TestClient) -> None:
        resp = client.get("/alerts?resolved=false")
        assert resp.status_code == 200


class TestResolveAlert:
    def test_resolve_existing_alert(self, client: TestClient) -> None:
        resp = client.put("/alerts/1/resolve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "resolved"

    def test_resolve_nonexistent_alert(self, client: TestClient) -> None:
        resp = client.put("/alerts/9999/resolve")
        assert resp.status_code == 404


class TestModelStatus:
    def test_get_model_status(self, client: TestClient) -> None:
        resp = client.get("/models/status")
        assert resp.status_code == 200
        data = resp.json()
        names = [m["model_name"] for m in data]
        assert "isolation_forest" in names
        assert "autoencoder" in names


class TestSensorHealth:
    def test_get_sensor_health(self, client: TestClient) -> None:
        resp = client.get("/sensors/health")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert "sensor_id" in data[0]
        assert "anomaly_rate" in data[0]
