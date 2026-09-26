"""Root cause analysis and explainability module for Industrial IoT anomaly detection.

Identifies top contributing sensors, computes precise physical deviations from normal
baselines, determines deviation direction, incorporates cross-sensor correlations,
evaluates multi-model agreement, and provides conservative, evidence-based explanations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SensorDeviation:
    """Detailed deviation metrics for a single contributing sensor.

    Attributes:
        sensor_id: Unique sensor identifier (e.g. ``pump_flow_00``).
        sensor_name: Base sensor name (e.g. ``pump_flow``).
        sensor_group: Subsystem group (e.g. ``flow``, ``pressure``).
        current_value: Measured value during the anomaly event.
        normal_reference_value: Expected reference value under normal operation.
        absolute_deviation: Magnitude of difference (|current - normal|).
        normalized_deviation: Deviation scaled by normal standard deviation (z-score like).
        percentage_deviation: Relative percentage deviation from reference.
        direction: Qualitative direction (``"Above normal"``, ``"Below normal"``, ``"Unstable / Noisy"``).
        recent_trend: Observed trend over the recent window (e.g. ``"Increasing (+1.4/min)"``).
        contribution_score: Normalized contribution score to overall anomaly.
        rank: Importance rank among all contributing sensors.
    """

    sensor_id: str
    sensor_name: str
    sensor_group: str
    current_value: float
    normal_reference_value: float
    absolute_deviation: float
    normalized_deviation: float
    percentage_deviation: float
    direction: str
    recent_trend: str
    contribution_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert deviation to dictionary."""
        return asdict(self)


@dataclass
class AnomalyExplanation:
    """Comprehensive explainability record for a single detected anomaly event.

    Answers:
        1. When did the anomaly happen? (timestamp)
        2. What was the ensemble score and threshold? (anomaly_score, threshold, detected)
        3. Which sensors contributed most? (top_contributing_sensors)
        4. How abnormal were they and in what direction? (sensor_deviations, deviation_direction)
        5. What was the suspected pattern? (anomaly_type, trend_information)
        6. Which models agreed? (model_scores, model_agreement)
        7. How does it relate to correlated process sensors? (correlated_sensors)
    """

    timestamp: Any
    anomaly_score: float
    threshold: float
    detected: bool
    anomaly_type: str
    top_contributing_sensors: list[SensorDeviation] = field(default_factory=list)
    sensor_deviations: dict[str, float] = field(default_factory=dict)
    sensor_values: dict[str, float] = field(default_factory=dict)
    normal_reference_values: dict[str, float] = field(default_factory=dict)
    deviation_direction: dict[str, str] = field(default_factory=dict)
    model_scores: dict[str, float] = field(default_factory=dict)
    model_agreement: dict[str, Any] = field(default_factory=dict)
    correlated_sensors: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    trend_information: dict[str, str] = field(default_factory=dict)
    explanation_text: str = ""
    narrative: str = ""
    ground_truth_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entire explanation to dictionary representation."""
        data = asdict(self)
        if isinstance(self.timestamp, (pd.Timestamp, datetime)):
            data["timestamp"] = self.timestamp.isoformat()
        else:
            data["timestamp"] = str(self.timestamp)
        return data


@dataclass
class RootCauseResult:
    """Legacy container for backwards compatibility with earlier audit tests.

    Attributes:
        timestamp: When the anomaly occurred.
        anomaly_score: Overall ensemble anomaly score.
        top_contributors: Ranked ``(sensor_id, contribution_score)`` pairs.
        all_contributions: Full contribution mapping.
        explanation: Human-readable summary.
    """

    timestamp: Any
    anomaly_score: float
    top_contributors: list[tuple[str, float]] = field(default_factory=list)
    all_contributions: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class RootCauseAnalyzer:
    """Analyze and explain detected Industrial IoT anomalies with physical context.

    Args:
        sensor_names: List of known sensor identifiers.
        sensor_configs: Optional mapping of sensor names to config dataclasses.
        normal_stats: Optional pre-computed normal reference statistics.
        correlation_matrix: Optional pre-computed pairwise correlation DataFrame.
        threshold: Ensemble detection threshold (default: 0.6061).
    """

    def __init__(
        self,
        sensor_names: list[str],
        sensor_configs: dict[str, Any] | None = None,
        normal_stats: dict[str, dict[str, float]] | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        threshold: float = 0.6061,
    ) -> None:
        self.sensor_names = sensor_names
        self.sensor_configs = sensor_configs or {}
        self.normal_stats = normal_stats or {}
        self.correlation_matrix = correlation_matrix
        self.threshold = threshold

    def compute_reference_stats(self, normal_df: pd.DataFrame) -> None:
        """Compute baseline normal reference statistics from training data.

        Args:
            normal_df: Wide-format DataFrame of normal training samples.
        """
        self.normal_stats = {}
        for col in normal_df.columns:
            if col in ("timestamp", "is_anomaly"):
                continue
            series = pd.to_numeric(normal_df[col], errors="coerce").dropna()
            if not series.empty:
                self.normal_stats[col] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()) if series.std() > 1e-6 else 1.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

    def compute_correlation_matrix(self, df: pd.DataFrame) -> None:
        """Compute and store pairwise Pearson correlations across sensors.

        Args:
            df: Wide-format DataFrame of sensor readings.
        """
        numeric_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
        self.correlation_matrix = df[numeric_cols].corr()

    def find_correlated_sensors(
        self,
        sensor_id: str,
        top_k: int = 3,
        min_corr: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Identify process variables strongly correlated with *sensor_id*.

        Args:
            sensor_id: Target sensor identifier.
            top_k: Maximum number of correlated sensors to return.
            min_corr: Minimum absolute correlation threshold.

        Returns:
            List of dictionaries with correlated sensor ID, Pearson r, and relationship type.
        """
        if self.correlation_matrix is None or sensor_id not in self.correlation_matrix.columns:
            return []

        corrs = self.correlation_matrix[sensor_id].drop(index=sensor_id, errors="ignore")
        # Sort by absolute correlation
        strong = corrs[corrs.abs() >= min_corr].sort_values(key=abs, ascending=False)

        results: list[dict[str, Any]] = []
        for other_id, r in strong.head(top_k).items():
            results.append(
                {
                    "sensor_id": str(other_id),
                    "sensor": str(other_id),
                    "correlation": float(r),
                    "relationship": "Positive co-variation" if r > 0 else "Inverse co-variation",
                }
            )
        return results

    def _infer_pattern(
        self,
        top_devs: list[SensorDeviation],
        recent_window: dict[str, list[float]] | None = None,
        ground_truth_type: str | None = None,
    ) -> dict[str, str]:
        """Infer observed anomaly pattern conservatively based on trajectory evidence."""
        if ground_truth_type:
            pattern = ground_truth_type.replace("_", " ").title()
            return {
                "observed_pattern": pattern,
                "evidence": f"Confirmed event pattern matching {pattern.lower()} profile.",
            }

        if not top_devs:
            return {
                "observed_pattern": "Unspecified deviation",
                "evidence": "General statistical deviation across sensor channels.",
            }

        primary = top_devs[0]
        s_id = primary.sensor_id

        # Analyze trajectory if recent window is supplied
        if recent_window and s_id in recent_window and len(recent_window[s_id]) >= 3:
            vals = np.array(recent_window[s_id])
            diffs = np.diff(vals)
            monotonic_up = np.all(diffs >= -1e-5)
            monotonic_down = np.all(diffs <= 1e-5)
            std_recent = float(np.std(vals))

            if np.all(vals == vals[0]):
                return {
                    "observed_pattern": "Stuck Sensor",
                    "evidence": "Sensor signal exhibits zero variability across successive observations.",
                }
            if monotonic_up or monotonic_down:
                direction = "upward" if monotonic_up else "downward"
                return {
                    "observed_pattern": "Gradual Drift",
                    "evidence": f"Observed monotonic {direction} drift away from the operating baseline.",
                }
            if std_recent > 3.0 * self.normal_stats.get(s_id, {}).get("std", 1.0):
                return {
                    "observed_pattern": "Increased Noise",
                    "evidence": "Signal variance significantly exceeds the normal operating noise envelope.",
                }

        # Multi-sensor check
        if len(top_devs) >= 3 and all(d.contribution_score > 0.4 for d in top_devs[:3]):
            return {
                "observed_pattern": "Correlated Multi-Sensor Deviation",
                "evidence": f"Synchronous deviations observed across {len(top_devs)} related process sensors.",
            }

        # Default classification based on normalized deviation
        if abs(primary.normalized_deviation) > 4.0:
            direction = "positive surge" if primary.normalized_deviation > 0 else "sharp drop"
            return {
                "observed_pattern": "Sudden Spike / Step",
                "evidence": f"Abrupt {direction} exceeding 4 standard deviations from reference.",
            }

        return {
            "observed_pattern": "Persistent Bias / Deviation",
            "evidence": f"Observed shift of {primary.absolute_deviation:.2f} units from normal reference.",
        }

    def explain_anomaly(
        self,
        timestamp: Any,
        anomaly_score: float,
        feature_contributions: dict[str, float] | None = None,
        current_values: dict[str, float] | None = None,
        model_scores: dict[str, float] | None = None,
        threshold: float | None = None,
        ground_truth_type: str | None = None,
        recent_window: dict[str, list[float]] | None = None,
        top_k: int = 3,
    ) -> AnomalyExplanation:
        """Construct a complete, evidence-based anomaly explanation.

        Args:
            timestamp: Timestamp or index of anomaly occurrence.
            anomaly_score: Overall ensemble anomaly score.
            feature_contributions: Mapping of sensor ID to contribution score in [0, 1].
            current_values: Mapping of sensor ID to raw measured value.
            model_scores: Optional individual detector scores.
            threshold: Optional score threshold (defaults to self.threshold).
            ground_truth_type: Optional injected ground truth label if known.
            recent_window: Optional dict of recent observations per sensor for trend detection.
            top_k: Number of primary contributors to highlight.

        Returns:
            Structured :class:`AnomalyExplanation` object.
        """
        thresh = threshold if threshold is not None else self.threshold
        is_detected = bool(anomaly_score > thresh)
        feat_contribs = feature_contributions or {}
        curr_vals = current_values or {}

        # Sort sensors by contribution score
        sorted_sensors = sorted(
            feat_contribs.items(), key=lambda item: item[1], reverse=True
        )

        sensor_deviations: dict[str, float] = {}
        normal_references: dict[str, float] = {}
        directions: dict[str, str] = {}
        top_deviations: list[SensorDeviation] = []

        for rank, (sensor_id, contrib) in enumerate(sorted_sensors, start=1):
            curr_val = curr_vals.get(sensor_id, 0.0)
            stats = self.normal_stats.get(sensor_id, {})
            norm_ref = stats.get("median", stats.get("mean", curr_val))
            std_ref = stats.get("std", 1.0)
            if std_ref <= 1e-6:
                std_ref = 1.0

            abs_dev = abs(curr_val - norm_ref)
            norm_dev = (curr_val - norm_ref) / std_ref
            pct_dev = (abs_dev / (abs(norm_ref) + 1e-5)) * 100.0

            if abs_dev < 0.5 * std_ref:
                direction = "Nominal"
            elif curr_val > norm_ref:
                direction = "Above Normal"
            else:
                direction = "Below Normal"

            # Trend over recent window
            trend_str = "Stable"
            if recent_window and sensor_id in recent_window and len(recent_window[sensor_id]) >= 2:
                series = recent_window[sensor_id]
                delta = series[-1] - series[0]
                if abs(delta) > 0.2 * std_ref:
                    direction_word = "Increasing" if delta > 0 else "Decreasing"
                    trend_str = f"{direction_word} ({delta:+.2f})"

            sensor_deviations[sensor_id] = float(abs_dev)
            normal_references[sensor_id] = float(norm_ref)
            directions[sensor_id] = direction

            base_name = sensor_id.rsplit("_", 1)[0] if "_" in sensor_id else sensor_id
            group = self.sensor_configs.get(base_name, None)
            group_name = getattr(group, "group", "general") if group else "process"

            if rank <= top_k:
                top_deviations.append(
                    SensorDeviation(
                        sensor_id=sensor_id,
                        sensor_name=base_name,
                        sensor_group=group_name,
                        current_value=float(curr_val),
                        normal_reference_value=float(norm_ref),
                        absolute_deviation=float(abs_dev),
                        normalized_deviation=float(norm_dev),
                        percentage_deviation=float(pct_dev),
                        direction=direction,
                        recent_trend=trend_str,
                        contribution_score=float(contrib),
                        rank=rank,
                    )
                )

        # Model Agreement analysis
        model_scores_dict = model_scores or {}
        agreed_models: list[str] = []
        details: dict[str, Any] = {}
        for m_name, m_score in model_scores_dict.items():
            # Individual threshold assumption: 0.5 or model score indicates outlier
            m_anom = bool(m_score > 0.5)
            if m_anom:
                agreed_models.append(m_name)
            details[m_name] = {
                "score": float(m_score),
                "is_anomaly": m_anom,
                "status": "Anomalous" if m_anom else "Normal",
            }

        total_models = max(1, len(model_scores_dict))
        agreement = {
            "agreed_models": agreed_models,
            "agreement_count": len(agreed_models),
            "total_models": len(model_scores_dict),
            "agreement_rate": len(agreed_models) / total_models,
            "details": details,
        }

        # Correlated sensors lookup for top contributors
        correlated_map: dict[str, list[dict[str, Any]]] = {}
        for dev in top_deviations:
            correlated_map[dev.sensor_id] = self.find_correlated_sensors(dev.sensor_id, top_k=3)

        # Pattern inference
        trend_info = self._infer_pattern(top_deviations, recent_window, ground_truth_type)

        # Build conservative, evidence-based narrative
        narrative_parts = []
        for dev in top_deviations:
            narrative_parts.append(
                f"{dev.sensor_id} ({dev.direction.lower()}, current: {dev.current_value:.2f}, "
                f"ref: {dev.normal_reference_value:.2f}, dev: {dev.absolute_deviation:+.2f})"
            )
        contributors_text = "; ".join(narrative_parts) if narrative_parts else "multiple sensors"

        severity_label = "Elevated" if anomaly_score > 0.8 else "Moderate"
        explanation_text = (
            f"Detected anomaly (score: {anomaly_score:.2f}, threshold: {thresh:.2f}). "
            f"Observed pattern: {trend_info['observed_pattern']}. "
            f"Primary contributing sensors: {contributors_text}. "
            f"Model agreement: {len(agreed_models)}/{len(model_scores_dict)} models flagged."
        )

        return AnomalyExplanation(
            timestamp=timestamp,
            anomaly_score=float(anomaly_score),
            threshold=float(thresh),
            detected=is_detected,
            anomaly_type=trend_info["observed_pattern"],
            top_contributing_sensors=top_deviations,
            sensor_deviations=sensor_deviations,
            sensor_values={k: float(v) for k, v in curr_vals.items()},
            normal_reference_values=normal_references,
            deviation_direction=directions,
            model_scores=model_scores_dict,
            model_agreement=agreement,
            correlated_sensors=correlated_map,
            trend_information=trend_info,
            explanation_text=explanation_text,
            narrative=explanation_text,
            ground_truth_type=ground_truth_type,
        )

    # -----------------------------------------------------------------------
    # Backwards-compatible methods from Phase 1
    # -----------------------------------------------------------------------

    def analyze(
        self,
        feature_contributions: dict[str, np.ndarray],
        anomaly_scores: np.ndarray,
        timestamps: np.ndarray | None,
        top_k: int = 3,
    ) -> list[RootCauseResult]:
        """Produce legacy root-cause results for all anomalous points.

        Args:
            feature_contributions: Per-sensor score arrays keyed by sensor name.
            anomaly_scores: Ensemble scores for every sample.
            timestamps: Aligned timestamp array (may be ``None``).
            top_k: Number of top contributors to include per anomaly.

        Returns:
            List of :class:`RootCauseResult` for anomalous samples only.
        """
        results: list[RootCauseResult] = []
        anomaly_indices = np.where(anomaly_scores > 0.5)[0]

        for idx in anomaly_indices:
            contributions = {
                sensor: float(scores[idx]) for sensor, scores in feature_contributions.items()
            }
            sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            top_contributors = sorted_contrib[:top_k]
            explanation = self._generate_explanation(top_contributors, anomaly_scores[idx])

            results.append(
                RootCauseResult(
                    timestamp=timestamps[idx] if timestamps is not None else idx,
                    anomaly_score=float(anomaly_scores[idx]),
                    top_contributors=top_contributors,
                    all_contributions=contributions,
                    explanation=explanation,
                )
            )

        logger.info("Root-cause analysis produced %d results", len(results))
        return results

    def _generate_explanation(
        self,
        top_contributors: list[tuple[str, float]],
        anomaly_score: float,
    ) -> str:
        """Build legacy summary string."""
        if not top_contributors:
            return "No significant contributors identified."

        parts: list[str] = []
        for sensor, score in top_contributors:
            sensor_type = sensor.split("_")[0] if "_" in sensor else sensor
            if score > 0.8:
                parts.append(f"{sensor_type} ({sensor}) showing critical deviation")
            elif score > 0.5:
                parts.append(f"{sensor_type} ({sensor}) showing moderate deviation")
            else:
                parts.append(f"{sensor_type} ({sensor}) showing minor deviation")

        severity = (
            "Critical" if anomaly_score > 0.8 else "Moderate" if anomaly_score > 0.5 else "Minor"
        )
        return f"{severity} anomaly detected. Primary contributors: {'; '.join(parts)}."

    def get_sensor_anomaly_rates(
        self,
        feature_contributions: dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Fraction of readings above *threshold* per sensor."""
        rates: dict[str, float] = {}
        for sensor, scores in feature_contributions.items():
            anomaly_count = int(np.sum(scores > threshold))
            rates[sensor] = anomaly_count / len(scores)
        return rates

    def get_temporal_patterns(
        self,
        feature_contributions: dict[str, np.ndarray],
        timestamps: np.ndarray,
        window_hours: int = 24,
    ) -> dict[str, dict[int, float]]:
        """Average anomaly score by hour-of-day for each sensor."""
        hours = pd.to_datetime(timestamps).hour

        patterns: dict[str, dict[int, float]] = {}
        for sensor, scores in feature_contributions.items():
            hourly_rates: dict[int, float] = {}
            for hour in range(24):
                mask = hours == hour
                if mask.sum() > 0:
                    hourly_rates[hour] = float(np.mean(scores[mask]))
            patterns[sensor] = hourly_rates

        return patterns
