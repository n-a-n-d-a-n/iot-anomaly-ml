"""Configurable alerting rules engine with cooldown and deduplication."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Alert:
    """A single alert raised by the rules engine.

    Attributes:
        timestamp: When the alert was generated.
        alert_type: Kind of rule that fired (``threshold``, ``frequency``, ``combination``).
        severity: One of ``low``, ``medium``, ``high``, ``critical``.
        message: Human-readable description.
        sensor_id: Optional sensor that triggered the alert.
        anomaly_score: Optional score value.
        metadata: Additional key-value data.
    """

    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    sensor_id: str | None = None
    anomaly_score: float | None = None
    metadata: dict | None = None


@dataclass
class AlertRule:
    """Definition of a single alerting rule.

    Attributes:
        name: Unique rule name.
        rule_type: ``threshold``, ``frequency``, or ``combination``.
        config: Rule-specific parameters.
        severity: Default alert severity.
        enabled: Whether the rule is active.
    """

    name: str
    rule_type: str
    config: dict = field(default_factory=dict)
    severity: str = "medium"
    enabled: bool = True


class AlertRulesEngine:
    """Evaluate incoming scoring results against a set of alert rules.

    Args:
        db_path: SQLite database path for persisting alerts.
        cooldown_seconds: Minimum seconds between repeated alerts for the same
            rule/sensor combination.
    """

    def __init__(self, db_path: str, cooldown_seconds: int = 300) -> None:
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        self.rules: list[AlertRule] = []
        self.recent_anomalies: defaultdict[str, list[datetime]] = defaultdict(list)
        self.last_alert_time: dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule.

        Args:
            rule: The rule to add.
        """
        self.rules.append(rule)

    def add_default_rules(self, config: dict[str, Any]) -> None:
        """Populate default rules from config values.

        Args:
            config: Alerting section of the application config.
        """
        freq = config.get("frequency_threshold", {})
        self.rules = [
            AlertRule(
                name="high_score",
                rule_type="threshold",
                config={"score_threshold": config.get("score_threshold", 0.8)},
                severity="high",
            ),
            AlertRule(
                name="frequent_anomalies",
                rule_type="frequency",
                config={
                    "count": freq.get("count", 5),
                    "minutes": freq.get("minutes", 10),
                },
                severity="critical",
            ),
            AlertRule(
                name="critical_sensor",
                rule_type="combination",
                config={
                    "sensor_types": ["temperature", "pressure"],
                    "score_threshold": 0.7,
                },
                severity="critical",
            ),
        ]

    def _check_cooldown(self, rule_name: str, sensor_id: str) -> bool:
        """Return ``True`` if the rule/sensor pair is still in cooldown."""
        key = f"{rule_name}:{sensor_id}"
        if key in self.last_alert_time:
            elapsed = (datetime.now() - self.last_alert_time[key]).total_seconds()
            return elapsed < self.cooldown_seconds
        return False

    def _update_cooldown(self, rule_name: str, sensor_id: str) -> None:
        key = f"{rule_name}:{sensor_id}"
        self.last_alert_time[key] = datetime.now()

    def evaluate(self, scoring_result: Any) -> list[Alert]:
        """Evaluate all rules against a scoring result.

        Args:
            scoring_result: An object with ``is_anomaly``, ``sensor_id``,
                ``timestamp``, and ``anomaly_score`` attributes.

        Returns:
            List of triggered alerts (may be empty).
        """
        alerts: list[Alert] = []

        if scoring_result.is_anomaly:
            self.recent_anomalies[scoring_result.sensor_id].append(scoring_result.timestamp)
            cutoff = datetime.now() - timedelta(minutes=30)
            self.recent_anomalies[scoring_result.sensor_id] = [
                t for t in self.recent_anomalies[scoring_result.sensor_id] if t > cutoff
            ]

        for rule in self.rules:
            if not rule.enabled:
                continue
            alert = self._evaluate_rule(rule, scoring_result)
            if alert and not self._check_cooldown(rule.name, scoring_result.sensor_id):
                alerts.append(alert)
                self._update_cooldown(rule.name, scoring_result.sensor_id)
                self._log_alert(alert)

        return alerts

    def _evaluate_rule(self, rule: AlertRule, result: Any) -> Alert | None:
        """Evaluate a single rule; return an Alert or None."""
        if rule.rule_type == "threshold":
            if result.anomaly_score >= rule.config["score_threshold"]:
                return Alert(
                    timestamp=result.timestamp,
                    alert_type="threshold",
                    severity=rule.severity,
                    message=f"High anomaly score ({result.anomaly_score:.2f}) on {result.sensor_id}",
                    sensor_id=result.sensor_id,
                    anomaly_score=result.anomaly_score,
                )

        elif rule.rule_type == "frequency":
            recent = self.recent_anomalies[result.sensor_id]
            cutoff = datetime.now() - timedelta(minutes=rule.config["minutes"])
            count = sum(1 for t in recent if t > cutoff)
            if count >= rule.config["count"]:
                return Alert(
                    timestamp=result.timestamp,
                    alert_type="frequency",
                    severity=rule.severity,
                    message=(
                        f"{count} anomalies in {rule.config['minutes']} minutes "
                        f"on {result.sensor_id}"
                    ),
                    sensor_id=result.sensor_id,
                    metadata={"count": count},
                )

        elif rule.rule_type == "combination":
            sensor_type = result.sensor_id.split("_")[0]
            if (
                sensor_type in rule.config["sensor_types"]
                and result.anomaly_score >= rule.config["score_threshold"]
            ):
                return Alert(
                    timestamp=result.timestamp,
                    alert_type="combination",
                    severity=rule.severity,
                    message=(
                        f"Critical sensor {result.sensor_id} anomaly "
                        f"(score: {result.anomaly_score:.2f})"
                    ),
                    sensor_id=result.sensor_id,
                    anomaly_score=result.anomaly_score,
                )

        return None

    def _log_alert(self, alert: Alert) -> None:
        """Persist alert to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO alert_history
               (timestamp, alert_type, severity, message, resolved)
               VALUES (?, ?, ?, ?, 0)""",
            (alert.timestamp.isoformat(), alert.alert_type, alert.severity, alert.message),
        )
        conn.commit()
        conn.close()
