"""SQLite database initialization and schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """\
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    sensor_type TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    value REAL NOT NULL,
    is_anomaly_ground_truth INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS anomaly_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    sensor_type TEXT,
    sensor_id TEXT,
    anomaly_score REAL NOT NULL,
    detection_method TEXT NOT NULL,
    contributing_features TEXT,
    resolved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS alert_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    anomaly_id INTEGER REFERENCES anomaly_log(id),
    resolved INTEGER DEFAULT 0,
    resolution_timestamp DATETIME
);

CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_log(timestamp);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialise the SQLite database, creating tables if needed.

    Args:
        db_path: Filesystem path for the SQLite database file.
            Parent directories are created automatically.

    Returns:
        An open :class:`sqlite3.Connection` to the database.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
