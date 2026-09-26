"""Tests for upgraded explainability, root-cause analysis, and correlation-aware diagnostics."""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.analysis.root_cause import (
    AnomalyExplanation,
    RootCauseAnalyzer,
    SensorDeviation,
)


@pytest.fixture()
def sample_sensor_data() -> pd.DataFrame:
    """Synthetic normal and anomalous sensor dataset."""
    rng = np.random.default_rng(42)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="min")
    
    # Strongly correlated pairs: flow and pressure, pump_speed and flow
    pump_speed = rng.normal(1500, 50, n)
    pump_flow = 0.05 * pump_speed + rng.normal(0, 1, n)
    pump_pressure = 0.8 * pump_flow + rng.normal(0, 0.5, n)
    temperature = rng.normal(60, 2, n)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "pump_speed": pump_speed,
        "pump_flow": pump_flow,
        "pump_pressure": pump_pressure,
        "temperature": temperature,
        "is_anomaly": [0] * n,
    })
    return df


@pytest.fixture()
def analyzer(sample_sensor_data: pd.DataFrame) -> RootCauseAnalyzer:
    sensors = ["pump_speed", "pump_flow", "pump_pressure", "temperature"]
    rca = RootCauseAnalyzer(sensor_names=sensors, threshold=0.6061)
    rca.compute_reference_stats(sample_sensor_data)
    rca.compute_correlation_matrix(sample_sensor_data)
    return rca


class TestReferenceStatsAndCorrelations:
    def test_compute_reference_stats(self, analyzer: RootCauseAnalyzer) -> None:
        assert "pump_speed" in analyzer.normal_stats
        assert "pump_flow" in analyzer.normal_stats
        stat = analyzer.normal_stats["pump_speed"]
        assert 1450 < stat["mean"] < 1550
        assert stat["std"] > 0

    def test_compute_correlation_matrix(self, analyzer: RootCauseAnalyzer) -> None:
        assert analyzer.correlation_matrix is not None
        assert "pump_flow" in analyzer.correlation_matrix.columns
        # pump_flow and pump_speed should be strongly positively correlated (> 0.8)
        r = analyzer.correlation_matrix.loc["pump_flow", "pump_speed"]
        assert r > 0.80

    def test_find_correlated_sensors(self, analyzer: RootCauseAnalyzer) -> None:
        corrs = analyzer.find_correlated_sensors("pump_flow", top_k=2, min_corr=0.4)
        assert len(corrs) >= 1
        sensor_ids = [c["sensor_id"] for c in corrs]
        assert "pump_speed" in sensor_ids or "pump_pressure" in sensor_ids
        for c in corrs:
            assert abs(c["correlation"]) >= 0.40
            assert "relationship" in c


class TestExplanationGeneration:
    def test_explain_anomaly_returns_valid_object(self, analyzer: RootCauseAnalyzer) -> None:
        now = datetime.now()
        curr_vals = {
            "pump_speed": 1500.0,
            "pump_flow": 120.0,  # Highly elevated above normal ~75
            "pump_pressure": 95.0,
            "temperature": 60.0,
        }
        contribs = {"pump_flow": 0.85, "pump_pressure": 0.65, "pump_speed": 0.1, "temperature": 0.05}
        model_scores = {
            "isolation_forest": 0.75,
            "autoencoder": 0.88,
            "dbscan": 0.62,
            "gmm": 0.70,
        }

        explanation = analyzer.explain_anomaly(
            timestamp=now,
            anomaly_score=0.785,
            feature_contributions=contribs,
            current_values=curr_vals,
            model_scores=model_scores,
            threshold=0.6061,
            ground_truth_type="sudden_spike",
        )

        assert isinstance(explanation, AnomalyExplanation)
        assert explanation.detected is True
        assert explanation.anomaly_score == 0.785
        assert explanation.threshold == 0.6061
        assert len(explanation.top_contributing_sensors) >= 1
        assert explanation.ground_truth_type == "sudden_spike"

    def test_top_sensor_ranking_and_deviations(self, analyzer: RootCauseAnalyzer) -> None:
        curr_vals = {
            "pump_speed": 1500.0,
            "pump_flow": 120.0,
            "pump_pressure": 50.0,
            "temperature": 60.0,
        }
        contribs = {"pump_flow": 0.95, "pump_pressure": 0.40}

        explanation = analyzer.explain_anomaly(
            timestamp=datetime.now(),
            anomaly_score=0.82,
            feature_contributions=contribs,
            current_values=curr_vals,
            top_k=3,
        )

        top1 = explanation.top_contributing_sensors[0]
        assert isinstance(top1, SensorDeviation)
        assert top1.sensor_id == "pump_flow"
        assert top1.direction == "Above Normal"
        assert top1.absolute_deviation > 0
        assert top1.normalized_deviation > 0
        assert top1.rank == 1

    def test_model_agreement_breakdown(self, analyzer: RootCauseAnalyzer) -> None:
        model_scores = {
            "isolation_forest": 0.75,
            "autoencoder": 0.85,
            "dbscan": 0.40,  # Below individual threshold
            "gmm": 0.65,
        }

        explanation = analyzer.explain_anomaly(
            timestamp=datetime.now(),
            anomaly_score=0.72,
            model_scores=model_scores,
            threshold=0.6061,
        )

        agreement = explanation.model_agreement
        assert agreement["agreement_count"] == 3
        assert "isolation_forest" in agreement["agreed_models"]
        assert "autoencoder" in agreement["agreed_models"]
        assert "gmm" in agreement["agreed_models"]
        assert "dbscan" not in agreement["agreed_models"]

    def test_conservative_narrative_wording(self, analyzer: RootCauseAnalyzer) -> None:
        explanation = analyzer.explain_anomaly(
            timestamp=datetime.now(),
            anomaly_score=0.80,
            feature_contributions={"pump_flow": 0.9},
            current_values={"pump_flow": 130.0},
        )
        narrative = explanation.narrative.lower()
        # Verify conservative, evidence-based wording
        assert "detected anomaly" in narrative
        # Must not claim equipment failure or damage
        assert "failed" not in narrative
        assert "damage" not in narrative
        assert "cyberattack" not in narrative

    def test_missing_or_empty_data_handling(self) -> None:
        empty_analyzer = RootCauseAnalyzer(sensor_names=[])
        # Calling explain_anomaly with empty data should not raise exception
        expl = empty_analyzer.explain_anomaly(
            timestamp=datetime.now(),
            anomaly_score=0.65,
            feature_contributions={},
            current_values={},
            model_scores={},
        )
        assert expl.detected is True
        assert len(expl.top_contributing_sensors) == 0
        assert expl.correlated_sensors == {}
