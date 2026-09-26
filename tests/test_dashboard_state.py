"""Tests for dashboard interactive state transitions, incident lifecycle, and actions."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.root_cause import RootCauseAnalyzer
from src.dashboard.state import Incident, MonitoringEngine


@pytest.fixture
def mock_telemetry_df() -> pd.DataFrame:
    base = datetime(2024, 1, 1, 0, 0, 0)
    sensors = ["pump_flow_00", "pump_pressure_00", "tank_1_level_00", "temperature_00"]
    rows = []
    for i in range(50):
        t = base + timedelta(minutes=i)
        is_anom = 1 if i in (10, 25, 40) else 0
        r = {"timestamp": t, "is_anomaly": is_anom}
        for s in sensors:
            r[s] = 85.0 + (i * 1.5 if is_anom else 0.2)
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture
def mock_analyzer(mock_telemetry_df: pd.DataFrame) -> RootCauseAnalyzer:
    cols = [c for c in mock_telemetry_df.columns if c not in ("timestamp", "is_anomaly")]
    analyzer = RootCauseAnalyzer(sensor_names=cols, threshold=0.6061)
    analyzer.compute_reference_stats(mock_telemetry_df)
    return analyzer


class TestMonitoringEngineStateTransitions:
    def test_initial_state(self) -> None:
        engine = MonitoringEngine(calibrated_threshold=0.6061, start_index=0)
        assert engine.monitoring_active is False
        assert engine.current_mode == "SETUP"
        assert engine.active_threshold == pytest.approx(0.6061)
        assert len(engine.incidents) == 0

    def test_start_and_pause_monitoring(self) -> None:
        engine = MonitoringEngine(start_index=0)
        engine.start_monitoring()
        assert engine.monitoring_active is True
        assert engine.current_mode == "LIVE MONITORING"

        engine.pause_monitoring()
        assert engine.monitoring_active is False
        # Mode remains LIVE MONITORING when paused
        assert engine.current_mode == "LIVE MONITORING"

    def test_reset_monitoring(self) -> None:
        engine = MonitoringEngine(start_index=5)
        engine.start_monitoring()
        engine.samples_processed = 42
        engine.anomalies_detected = 3
        engine.reset_monitoring(start_index=0)

        assert engine.monitoring_active is False
        assert engine.current_mode == "SETUP"
        assert engine.samples_processed == 0
        assert engine.anomalies_detected == 0
        assert engine.replay_index == 0


class TestIncidentLifecycle:
    def test_step_telemetry_detects_anomaly_and_creates_incident(
        self,
        mock_telemetry_df: pd.DataFrame,
        mock_analyzer: RootCauseAnalyzer,
    ) -> None:
        engine = MonitoringEngine(calibrated_threshold=0.6061, start_index=9)
        # Advance 2 steps: index 9 (normal), index 10 (anomaly)
        inc = engine.step_telemetry(mock_telemetry_df, mock_analyzer, n_steps=2)

        assert inc is not None
        assert inc.incident_id == "INC-001"
        assert inc.status == "NEW"
        assert inc.ensemble_score >= engine.active_threshold
        assert len(engine.incidents) == 1

    def test_incident_selection_and_status_transitions(
        self,
        mock_telemetry_df: pd.DataFrame,
        mock_analyzer: RootCauseAnalyzer,
    ) -> None:
        engine = MonitoringEngine(calibrated_threshold=0.6061, start_index=10)
        inc = engine.step_telemetry(mock_telemetry_df, mock_analyzer, n_steps=1)
        assert inc is not None

        # 1. Select incident transitions status from NEW -> INVESTIGATING
        selected = engine.select_incident(inc.incident_id)
        assert selected is True
        assert engine.current_mode == "INVESTIGATION"
        assert inc.status == "INVESTIGATING"

        # 2. Acknowledge incident
        ack = engine.acknowledge_incident(inc.incident_id)
        assert ack is True
        assert inc.status == "ACKNOWLEDGED"

        # 3. Mark for review
        rev = engine.mark_for_review(inc.incident_id)
        assert rev is True
        assert inc.status == "REVIEW"

        # 4. Resolve incident
        res = engine.resolve_incident(inc.incident_id)
        assert res is True
        assert inc.status == "RESOLVED"

    def test_jump_to_next_anomaly(
        self,
        mock_telemetry_df: pd.DataFrame,
        mock_analyzer: RootCauseAnalyzer,
    ) -> None:
        engine = MonitoringEngine(start_index=0)
        # Jump from 0 to next anomaly (which is at index 10)
        inc = engine.jump_to_next_anomaly(mock_telemetry_df, mock_analyzer)
        assert inc is not None
        assert inc.row_index == 10
        assert engine.replay_index == 11

    def test_operator_notes_and_report_generation(
        self,
        mock_telemetry_df: pd.DataFrame,
        mock_analyzer: RootCauseAnalyzer,
    ) -> None:
        engine = MonitoringEngine(start_index=10)
        inc = engine.step_telemetry(mock_telemetry_df, mock_analyzer, n_steps=1)
        assert inc is not None

        added = engine.add_operator_note(inc.incident_id, "Pump pressure spike verified with maintenance.", author="Alice")
        assert added is True
        assert len(inc.notes) == 1
        assert "Pump pressure spike" in inc.notes[0]["text"]

        report = engine.generate_incident_report(inc.incident_id)
        assert "INDUSTRIAL IoT ANOMALY AUDIT REPORT" in report
        assert inc.incident_id in report
        assert "Alice" in report
        assert "Pump pressure spike" in report
