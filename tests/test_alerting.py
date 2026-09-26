"""Tests for alerting rules engine and notification channels."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import pytest

from src.alerting.channels import AlertDispatcher, ConsoleChannel, FileChannel
from src.alerting.rules_engine import Alert, AlertRule, AlertRulesEngine
from src.utils.database import init_db


@dataclass
class _FakeScoringResult:
    timestamp: datetime
    sensor_id: str
    anomaly_score: float
    is_anomaly: bool


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture()
def engine(db_path: str) -> AlertRulesEngine:
    eng = AlertRulesEngine(db_path, cooldown_seconds=0)
    eng.add_default_rules(
        {
            "score_threshold": 0.8,
            "frequency_threshold": {"count": 3, "minutes": 10},
        }
    )
    return eng


class TestThresholdRule:
    def test_triggers_above_threshold(self, engine: AlertRulesEngine) -> None:
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="temperature_00",
            anomaly_score=0.9,
            is_anomaly=True,
        )
        alerts = engine.evaluate(result)
        threshold_alerts = [a for a in alerts if a.alert_type == "threshold"]
        assert len(threshold_alerts) >= 1

    def test_does_not_trigger_below_threshold(self, engine: AlertRulesEngine) -> None:
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="vibration_00",
            anomaly_score=0.3,
            is_anomaly=False,
        )
        alerts = engine.evaluate(result)
        threshold_alerts = [a for a in alerts if a.alert_type == "threshold"]
        assert len(threshold_alerts) == 0


class TestFrequencyRule:
    def test_triggers_after_n_anomalies(self, engine: AlertRulesEngine) -> None:
        for i in range(5):
            result = _FakeScoringResult(
                timestamp=datetime.now(),
                sensor_id="temperature_00",
                anomaly_score=0.5,
                is_anomaly=True,
            )
            alerts = engine.evaluate(result)
        freq_alerts = [a for a in alerts if a.alert_type == "frequency"]
        assert len(freq_alerts) >= 1


class TestCombinationRule:
    def test_triggers_for_critical_sensor(self, engine: AlertRulesEngine) -> None:
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="temperature_00",
            anomaly_score=0.9,
            is_anomaly=True,
        )
        alerts = engine.evaluate(result)
        combo_alerts = [a for a in alerts if a.alert_type == "combination"]
        assert len(combo_alerts) >= 1

    def test_does_not_trigger_for_non_critical_sensor(self, engine: AlertRulesEngine) -> None:
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="vibration_00",
            anomaly_score=0.9,
            is_anomaly=True,
        )
        alerts = engine.evaluate(result)
        combo_alerts = [a for a in alerts if a.alert_type == "combination"]
        assert len(combo_alerts) == 0


class TestCooldown:
    def test_cooldown_prevents_duplicates(self, db_path: str) -> None:
        engine = AlertRulesEngine(db_path, cooldown_seconds=60)
        engine.add_rule(
            AlertRule(
                name="test_rule",
                rule_type="threshold",
                config={"score_threshold": 0.5},
                severity="high",
            )
        )
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="temp_00",
            anomaly_score=0.9,
            is_anomaly=True,
        )
        alerts1 = engine.evaluate(result)
        alerts2 = engine.evaluate(result)
        # First should trigger, second should be suppressed
        assert len(alerts1) >= 1
        assert len(alerts2) == 0


class TestAlertLogging:
    def test_alert_persisted_to_db(self, engine: AlertRulesEngine, db_path: str) -> None:
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="temperature_00",
            anomaly_score=0.9,
            is_anomaly=True,
        )
        engine.evaluate(result)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0]
        conn.close()
        assert count >= 1


class TestDisabledRule:
    def test_disabled_rule_skipped(self, db_path: str) -> None:
        engine = AlertRulesEngine(db_path, cooldown_seconds=0)
        engine.add_rule(
            AlertRule(
                name="disabled",
                rule_type="threshold",
                config={"score_threshold": 0.1},
                severity="high",
                enabled=False,
            )
        )
        result = _FakeScoringResult(
            timestamp=datetime.now(),
            sensor_id="temp_00",
            anomaly_score=0.9,
            is_anomaly=True,
        )
        alerts = engine.evaluate(result)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Channel tests
# ---------------------------------------------------------------------------


class TestConsoleChannel:
    @pytest.mark.asyncio
    async def test_send_returns_true(self) -> None:
        channel = ConsoleChannel()
        alert = Alert(
            timestamp=datetime.now(),
            alert_type="threshold",
            severity="high",
            message="test alert",
        )
        assert await channel.send(alert) is True


class TestFileChannel:
    @pytest.mark.asyncio
    async def test_appends_to_file(self, tmp_path) -> None:
        path = str(tmp_path / "alerts.log")
        channel = FileChannel(path)
        alert = Alert(
            timestamp=datetime.now(),
            alert_type="threshold",
            severity="critical",
            message="sensor overheating",
        )
        await channel.send(alert)
        with open(path) as fh:
            content = fh.read()
        assert "CRITICAL" in content
        assert "sensor overheating" in content


class TestAlertDispatcher:
    @pytest.mark.asyncio
    async def test_dispatches_to_all_channels(self, tmp_path) -> None:
        dispatcher = AlertDispatcher()
        file1 = str(tmp_path / "a.log")
        file2 = str(tmp_path / "b.log")
        dispatcher.add_channel(FileChannel(file1))
        dispatcher.add_channel(FileChannel(file2))

        alert = Alert(
            timestamp=datetime.now(),
            alert_type="threshold",
            severity="high",
            message="test dispatch",
        )
        await dispatcher.dispatch(alert)

        with open(file1) as fh:
            assert "test dispatch" in fh.read()
        with open(file2) as fh:
            assert "test dispatch" in fh.read()
