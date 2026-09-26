"""Tests for SQLite database initialization and schema."""

from pathlib import Path

import pytest

from src.utils.database import init_db


@pytest.fixture()
def db_path(tmp_path):
    """Return a temporary database path."""
    return str(tmp_path / "test.db")


def test_init_db_creates_file(db_path: str) -> None:
    conn = init_db(db_path)
    assert Path(db_path).exists()
    conn.close()


def test_init_db_creates_sensor_readings_table(db_path: str) -> None:
    conn = init_db(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_readings'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_init_db_creates_anomaly_log_table(db_path: str) -> None:
    conn = init_db(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_log'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_init_db_creates_alert_history_table(db_path: str) -> None:
    conn = init_db(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alert_history'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_init_db_creates_indexes(db_path: str) -> None:
    conn = init_db(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    )
    indexes = [row[0] for row in cursor.fetchall()]
    assert "idx_readings_timestamp" in indexes
    assert "idx_anomaly_timestamp" in indexes
    conn.close()


def test_init_db_creates_parent_dirs(tmp_path) -> None:
    nested_path = str(tmp_path / "a" / "b" / "test.db")
    conn = init_db(nested_path)
    assert Path(nested_path).exists()
    conn.close()


def test_init_db_idempotent(db_path: str) -> None:
    conn1 = init_db(db_path)
    conn1.execute(
        "INSERT INTO sensor_readings (timestamp, sensor_type, sensor_id, value) "
        "VALUES ('2024-01-01', 'temp', 's1', 70.0)"
    )
    conn1.commit()
    conn1.close()

    conn2 = init_db(db_path)
    cursor = conn2.execute("SELECT COUNT(*) FROM sensor_readings")
    assert cursor.fetchone()[0] == 1
    conn2.close()


def test_sensor_readings_schema(db_path: str) -> None:
    conn = init_db(db_path)
    cursor = conn.execute("PRAGMA table_info(sensor_readings)")
    columns = {row[1] for row in cursor.fetchall()}
    expected = {"id", "timestamp", "sensor_type", "sensor_id", "value", "is_anomaly_ground_truth"}
    assert expected == columns
    conn.close()
