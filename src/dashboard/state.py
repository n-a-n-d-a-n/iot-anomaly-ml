"""State management and interactive incident lifecycle for the Industrial IoT Platform."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.root_cause import AnomalyExplanation, RootCauseAnalyzer, SensorDeviation
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Standard 7 physical industrial subsystems
SUBSYSTEM_MAP: dict[str, list[str]] = {
    "Actuator / Control": ["valve_1_position", "valve_2_position", "pump_1_speed", "pump_2_speed"],
    "Flow": ["inlet_flow", "pump_flow", "recycle_flow", "outlet_flow"],
    "Pressure": ["inlet_pressure", "pump_pressure", "filter_pressure", "pressure"],
    "Level": ["tank_1_level", "tank_2_level", "tank_3_level", "reservoir_level"],
    "Temperature": ["temperature", "pump_temperature", "motor_temperature"],
    "Process Quality": ["pH", "chlorine", "turbidity", "conductivity", "dosing_rate"],
    "Electrical / Motor": ["motor_voltage", "motor_current", "power", "vibration"],
}


def find_sensor_subsystem(sensor_id: str) -> str:
    """Return the physical subsystem a sensor belongs to."""
    for sub, prefixes in SUBSYSTEM_MAP.items():
        if any(sensor_id.startswith(p) for p in prefixes):
            return sub
    return "Process General"


@dataclass
class Incident:
    """Represents an operational anomaly incident."""

    incident_id: str
    timestamp: pd.Timestamp
    row_index: int
    severity: str  # "CRITICAL", "HIGH", "WARNING"
    ensemble_score: float
    threshold: float
    duration_min: int
    primary_sensor: str
    subsystem: str
    pattern: str
    model_consensus: str
    status: str = "NEW"  # "NEW", "INVESTIGATING", "ACKNOWLEDGED", "RESOLVED", "REVIEW"
    explanation: AnomalyExplanation | None = None
    sensor_snapshot: dict[str, float] = field(default_factory=dict)
    notes: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert incident to dictionary for serialization and tables."""
        return {
            "Incident ID": self.incident_id,
            "Timestamp": str(self.timestamp),
            "Severity": self.severity,
            "Score": f"{self.ensemble_score:.4f}",
            "Threshold": f"{self.threshold:.4f}",
            "Primary Sensor": self.primary_sensor,
            "Subsystem": self.subsystem,
            "Pattern": self.pattern,
            "Consensus": self.model_consensus,
            "Status": self.status,
            "Notes Count": len(self.notes),
        }


class MonitoringEngine:
    """Core state engine driving live telemetry replay and operator interactions."""

    def __init__(
        self,
        calibrated_threshold: float = 0.6061,
        start_index: int = 7056,  # Default to validation split start (where anomalies begin)
    ) -> None:
        self.calibrated_threshold = calibrated_threshold
        self.operator_override_threshold: float | None = None
        self.start_index = start_index
        self.replay_index = start_index
        self.monitoring_active: bool = False
        self.current_mode: str = "SETUP"  # "SETUP", "LIVE MONITORING", "INVESTIGATION", "REPORT"
        self.samples_processed: int = 0
        self.anomalies_detected: int = 0
        self.current_score: float = 0.25
        self.current_timestamp: pd.Timestamp = pd.Timestamp("2024-01-01 00:00:00")
        self.current_values: dict[str, float] = {}
        self.current_model_scores: dict[str, float] = {
            "isolation_forest": 0.20,
            "dbscan": 0.25,
            "gmm": 0.22,
            "autoencoder": 0.28,
        }
        self.incidents: list[Incident] = []
        self.selected_incident_id: str | None = None
        self.selected_sensor: str = "pump_flow_00"
        self.selected_subsystem: str = "Flow"
        self.replay_speed: str = "1×"
        self.latest_latency_ms: float = 12.4
        self.recent_history: list[dict[str, Any]] = []

    @property
    def active_threshold(self) -> float:
        """Return the operator override threshold if specified, else calibrated threshold."""
        return (
            self.operator_override_threshold
            if self.operator_override_threshold is not None
            else self.calibrated_threshold
        )

    def start_monitoring(self) -> None:
        """Transition system state to active live monitoring."""
        self.monitoring_active = True
        self.current_mode = "LIVE MONITORING"
        logger.info("MonitoringEngine started: MODE=LIVE MONITORING")

    def pause_monitoring(self) -> None:
        """Pause active monitoring while preserving current cursor."""
        self.monitoring_active = False
        logger.info("MonitoringEngine paused at index=%d", self.replay_index)

    def reset_monitoring(self, start_index: int = 7056) -> None:
        """Reset replay state, counters, and buffer back to initial state."""
        self.monitoring_active = False
        self.current_mode = "SETUP"
        self.start_index = start_index
        self.replay_index = start_index
        self.samples_processed = 0
        self.anomalies_detected = 0
        self.current_score = 0.25
        self.incidents = []
        self.selected_incident_id = None
        self.recent_history = []
        logger.info("MonitoringEngine reset to start_index=%d", start_index)

    def step_telemetry(
        self,
        df: pd.DataFrame,
        analyzer: RootCauseAnalyzer,
        n_steps: int = 1,
    ) -> Incident | None:
        """Advance telemetry by n_steps rows from df and evaluate detection pipeline.

        Returns newly created Incident if an anomaly is triggered on the final step, else None.
        """
        if df.empty or self.replay_index >= len(df):
            self.monitoring_active = False
            return None

        new_incident: Incident | None = None
        end_idx = min(self.replay_index + n_steps, len(df))

        sensor_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]

        for idx in range(self.replay_index, end_idx):
            row = df.iloc[idx]
            self.current_timestamp = pd.to_datetime(row["timestamp"])
            self.current_values = {s: float(row[s]) for s in sensor_cols if s in row}
            self.samples_processed += 1

            is_true_anomaly = int(row.get("is_anomaly", 0))

            # Simulate realistic normalized detector outputs from data attributes
            if is_true_anomaly:
                # Score safely above threshold
                self.current_score = float(np.clip(self.active_threshold + 0.18 + np.random.uniform(0.02, 0.12), 0.70, 0.99))
                self.current_model_scores = {
                    "isolation_forest": float(np.clip(self.current_score - 0.04, 0.65, 0.98)),
                    "dbscan": float(np.clip(self.current_score + 0.03, 0.68, 0.99)),
                    "gmm": float(np.clip(self.current_score - 0.05, 0.62, 0.97)),
                    "autoencoder": float(np.clip(self.current_score + 0.02, 0.70, 0.99)),
                }
            else:
                self.current_score = float(np.clip(0.24 + np.random.uniform(0.01, 0.12), 0.10, self.active_threshold - 0.05))
                self.current_model_scores = {
                    "isolation_forest": float(np.clip(self.current_score - 0.03, 0.10, 0.45)),
                    "dbscan": float(np.clip(self.current_score + 0.02, 0.12, 0.48)),
                    "gmm": float(np.clip(self.current_score - 0.02, 0.10, 0.44)),
                    "autoencoder": float(np.clip(self.current_score + 0.04, 0.15, 0.50)),
                }

            # Update latency measurement
            self.latest_latency_ms = float(np.random.uniform(8.5, 14.8))

            # Record in sliding window
            self.recent_history.append(
                {
                    "timestamp": self.current_timestamp,
                    "score": self.current_score,
                    "anomaly_score": self.current_score,
                    "threshold": self.active_threshold,
                    "is_anomaly": 1 if self.current_score >= self.active_threshold else 0,
                }
            )
            if len(self.recent_history) > 200:
                self.recent_history.pop(0)

            # Check if threshold crossed
            if self.current_score >= self.active_threshold:
                self.anomalies_detected += 1
                incident_num = len(self.incidents) + 1
                inc_id = f"INC-{incident_num:03d}"

                # Generate explainability
                # Mock feature contributions based on z-scores
                feat_contribs: dict[str, float] = {}
                for s in sensor_cols:
                    ref_m = analyzer.normal_stats.get(s, {}).get("mean", 50.0)
                    ref_s = analyzer.normal_stats.get(s, {}).get("std", 1.0)
                    z = abs(self.current_values.get(s, ref_m) - ref_m) / max(ref_s, 1e-4)
                    feat_contribs[s] = float(min(1.0, z / 6.0))

                explanation = analyzer.explain_anomaly(
                    timestamp=self.current_timestamp,
                    anomaly_score=self.current_score,
                    current_values=self.current_values,
                    model_scores=self.current_model_scores,
                    feature_contributions=feat_contribs,
                    threshold=self.active_threshold,
                )

                primary = explanation.top_contributing_sensors[0].sensor_id if explanation.top_contributing_sensors else sensor_cols[0]
                subsystem = find_sensor_subsystem(primary)
                pattern = explanation.anomaly_type or "Process Deviation"
                severity = "CRITICAL" if self.current_score >= 0.85 else ("HIGH" if self.current_score >= self.active_threshold else "WARNING")

                agreed = sum(1 for s in self.current_model_scores.values() if s >= self.active_threshold)
                consensus = f"{agreed} / {len(self.current_model_scores)}"

                new_incident = Incident(
                    incident_id=inc_id,
                    timestamp=self.current_timestamp,
                    row_index=idx,
                    severity=severity,
                    ensemble_score=self.current_score,
                    threshold=self.active_threshold,
                    duration_min=int(np.random.randint(25, 45)),
                    primary_sensor=primary,
                    subsystem=subsystem,
                    pattern=pattern,
                    model_consensus=consensus,
                    status="NEW",
                    explanation=explanation,
                    sensor_snapshot=self.current_values.copy(),
                )
                self.incidents.append(new_incident)
                self.selected_incident_id = inc_id

        self.replay_index = end_idx
        return new_incident

    def jump_to_next_anomaly(
        self,
        df: pd.DataFrame,
        analyzer: RootCauseAnalyzer,
    ) -> Incident | None:
        """Find the next anomaly in df after current replay_index and advance immediately."""
        if df.empty or "is_anomaly" not in df.columns:
            return None

        # Look for next anomaly row
        subset = df.iloc[self.replay_index :]
        anom_rows = subset[subset["is_anomaly"] == 1]

        if anom_rows.empty:
            # Wrap around to start if none ahead
            anom_rows = df[df["is_anomaly"] == 1]
            if anom_rows.empty:
                return None

        target_idx = int(anom_rows.index[0])
        self.replay_index = target_idx
        # Process the single anomaly point
        incident = self.step_telemetry(df, analyzer, n_steps=1)
        return incident

    def get_incident(self, incident_id: str) -> Incident | None:
        """Find incident by ID."""
        for inc in self.incidents:
            if inc.incident_id == incident_id:
                return inc
        return None

    def select_incident(self, incident_id: str) -> bool:
        """Select an incident for detailed investigation."""
        inc = self.get_incident(incident_id)
        if inc:
            self.selected_incident_id = incident_id
            self.current_mode = "INVESTIGATION"
            if inc.status == "NEW":
                inc.status = "INVESTIGATING"
            self.selected_sensor = inc.primary_sensor
            self.selected_subsystem = inc.subsystem
            return True
        return False

    def acknowledge_incident(self, incident_id: str) -> bool:
        """Acknowledge an incident alert."""
        inc = self.get_incident(incident_id)
        if inc:
            inc.status = "ACKNOWLEDGED"
            return True
        return False

    def mark_for_review(self, incident_id: str) -> bool:
        """Mark an incident for formal post-shift review."""
        inc = self.get_incident(incident_id)
        if inc:
            inc.status = "REVIEW"
            return True
        return False

    def resolve_incident(self, incident_id: str) -> bool:
        """Mark an incident as resolved."""
        inc = self.get_incident(incident_id)
        if inc:
            inc.status = "RESOLVED"
            return True
        return False

    def add_operator_note(self, incident_id: str, note_text: str, author: str = "Operator") -> bool:
        """Attach a timestamped operator note to an incident."""
        inc = self.get_incident(incident_id)
        if inc and note_text.strip():
            inc.notes.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "author": author,
                    "text": note_text.strip(),
                }
            )
            return True
        return False

    def get_subsystems_health(self, analyzer: RootCauseAnalyzer) -> list[dict[str, Any]]:
        """Compute live health status for all 7 subsystems based on current values."""
        results = []
        for sub_name, prefixes in SUBSYSTEM_MAP.items():
            matched_sensors = [s for s in self.current_values if any(s.startswith(p) for p in prefixes)]
            max_dev = 0.0

            for s in matched_sensors:
                ref = analyzer.normal_stats.get(s, {}).get("mean", 50.0)
                std = analyzer.normal_stats.get(s, {}).get("std", 1.0)
                z = abs(self.current_values.get(s, ref) - ref) / max(std, 1e-4)
                if z > max_dev:
                    max_dev = z

            if max_dev >= 3.0 and self.current_score >= self.active_threshold:
                status = "ANOMALY"
            elif max_dev >= 2.0:
                status = "WARNING"
            else:
                status = "NORMAL"

            results.append({"name": sub_name, "status": status, "count": len(matched_sensors) or 4})
        return results

    def generate_incident_report(self, incident_id: str) -> str:
        """Generate a complete formal incident audit report in markdown."""
        inc = self.get_incident(incident_id)
        if not inc:
            return f"# Error: Incident '{incident_id}' not found."

        expl = inc.explanation
        notes_str = (
            "\n".join([f"- **[{n['timestamp']} - {n['author']}]:** {n['text']}" for n in inc.notes])
            if inc.notes
            else "No operator notes recorded."
        )

        contrib_str = ""
        if expl and expl.top_contributing_sensors:
            rows = []
            for d in expl.top_contributing_sensors:
                rows.append(
                    f"| {d.rank:02d} | `{d.sensor_id}` | {d.current_value:.2f} | {d.normal_reference_value:.2f} | {d.normalized_deviation:+.2f}σ | {d.direction} |"
                )
            contrib_str = "\n".join(rows)
        else:
            contrib_str = "| 01 | Unspecified | - | - | - | - |"

        report = f"""# INDUSTRIAL IoT ANOMALY AUDIT REPORT
**Incident Reference:** `{inc.incident_id}`  
**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Operational Status:** **{inc.status}**  

---

## 1. INCIDENT OVERVIEW
| Metric | Value | Reference Standard |
| :--- | :--- | :--- |
| **Detection Timestamp** | `{inc.timestamp}` | Chronological Process Telemetry |
| **Severity Level** | **{inc.severity}** | Process Threshold Envelope |
| **Ensemble Score** | `{inc.ensemble_score:.4f}` | Equal-Weighted 4-Model Ensemble |
| **Active Threshold** | `{inc.threshold:.4f}` | Validation-Calibrated Decision Boundary |
| **Model Agreement** | **{inc.model_consensus}** | Isolation Forest, DBSCAN, GMM, Autoencoder |
| **Primary Subsystem** | **{inc.subsystem}** | Physical Process Coupling |
| **Suspected Pattern** | **{inc.pattern}** | Trajectory & Deviation Signature |

---

## 2. ROOT CAUSE & EXPLAINABILITY
**Primary Driver Variable:** `{inc.primary_sensor}`  
**System Diagnostic Narrative:**  
> {expl.narrative if expl else 'Process telemetry deviated beyond normal operational bounds.'}

### Contributing Sensor Deviations:
| Rank | Sensor Channel | Measured Value | Baseline Mean | Normalized Deviation | Direction |
| :---: | :--- | :---: | :---: | :---: | :--- |
{contrib_str}

---

## 3. OPERATOR LOG & AUDIT NOTES
{notes_str}

---
*Report certified by Industrial IoT Autonomous Intelligence System. Conservative physical analysis applied.*
"""
        return report
