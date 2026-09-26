"""Tests for dashboard reusable chart components."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard.components import (
    create_anomaly_timeline,
    create_contribution_chart,
    create_correlation_bar,
    create_health_gauge,
    create_model_agreement_chart,
    create_score_timeline,
    create_sensor_timeseries_view,
)


@pytest.fixture()
def sensor_df() -> pd.DataFrame:
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(50):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "sensor_id": "temp_00",
                "value": 70.0 + i * 0.1,
                "is_anomaly": 1 if i in (10, 20, 30) else 0,
            }
        )
    return pd.DataFrame(rows)


class TestCreateAnomalyTimeline:
    def test_returns_figure(self, sensor_df: pd.DataFrame) -> None:
        fig = create_anomaly_timeline(sensor_df)
        assert isinstance(fig, go.Figure)

    def test_has_anomaly_trace(self, sensor_df: pd.DataFrame) -> None:
        fig = create_anomaly_timeline(sensor_df)
        trace_names = [t.name for t in fig.data]
        assert "Anomalies" in trace_names

    def test_no_anomaly_trace_when_none(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "sensor_id": ["s1"],
                "value": [1.0],
                "is_anomaly": [0],
            }
        )
        fig = create_anomaly_timeline(df)
        trace_names = [t.name for t in fig.data]
        assert "Anomalies" not in trace_names


class TestCreateContributionChart:
    def test_returns_figure(self) -> None:
        fig = create_contribution_chart({"temp": 0.9, "vib": 0.3})
        assert isinstance(fig, go.Figure)

    def test_bar_count(self) -> None:
        fig = create_contribution_chart({"a": 0.5, "b": 0.2, "c": 0.8})
        bar_trace = fig.data[0]
        assert len(bar_trace.x) == 3


class TestCreateHealthGauge:
    def test_returns_figure(self) -> None:
        fig = create_health_gauge(0.03, "temp_00")
        assert isinstance(fig, go.Figure)

    def test_gauge_value(self) -> None:
        fig = create_health_gauge(0.05, "sensor")
        indicator = fig.data[0]
        assert indicator.value == pytest.approx(5.0)


class TestPhase2BChartComponents:
    def test_create_score_timeline(self, sensor_df: pd.DataFrame) -> None:
        df = sensor_df.copy()
        df["anomaly_score"] = 0.4
        df.loc[10, "anomaly_score"] = 0.85
        fig = create_score_timeline(df, threshold=0.6061)
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Ensemble Score" in names
        assert any("Threshold" in n for n in names)
        assert "Detected Anomalies" in names

    def test_create_sensor_timeseries_view(self, sensor_df: pd.DataFrame) -> None:
        fig = create_sensor_timeseries_view(
            sensor_df,
            sensor_col="value",
            ref_mean=72.0,
            ref_std=2.0,
            highlight_time=sensor_df["timestamp"].iloc[10],
        )
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert any("Envelope" in n for n in names)
        assert "Sensor Reading" in names
        assert "Anomaly Event" in names

    def test_create_model_agreement_chart(self) -> None:
        scores = {"isolation_forest": 0.7, "autoencoder": 0.8, "dbscan": 0.5, "gmm": 0.65}
        fig = create_model_agreement_chart(scores, threshold=0.6061)
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 4

    def test_create_correlation_bar(self) -> None:
        corrs = [
            {"sensor_id": "pump_speed", "correlation": 0.95},
            {"sensor_id": "pump_pressure", "correlation": 0.82},
        ]
        fig = create_correlation_bar(corrs, "pump_flow")
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 2


class TestPhase2CComponentsAndTheme:
    def test_create_hero_gauge(self) -> None:
        from src.dashboard.components import create_hero_gauge
        fig = create_hero_gauge(0.85, threshold=0.6061)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].value == pytest.approx(0.85)

    def test_create_radar_comparison_chart(self) -> None:
        from src.dashboard.components import create_radar_comparison_chart
        eval_dict = {
            "isolation_forest": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "auc_roc": 0.85, "pr_auc": 0.72},
            "ensemble": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "auc_roc": 0.95, "pr_auc": 0.88},
        }
        fig = create_radar_comparison_chart(eval_dict)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_create_correlation_heatmap(self) -> None:
        from src.dashboard.components import create_correlation_heatmap
        df = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["s1", "s2"], index=["s1", "s2"])
        fig = create_correlation_heatmap(df)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "heatmap"

    def test_theme_renderers(self) -> None:
        from src.dashboard.theme import render_app_header, render_plant_pulse, render_why_alert_panel
        header = render_app_header("2024-01-01")
        assert "INDUSTRIAL IoT ANOMALY INTELLIGENCE" in header
        assert "SYSTEM ONLINE" in header

        pulse = render_plant_pulse([{"name": "FLOW", "status": "NORMAL", "count": 4}])
        assert "FLOW" in pulse
        assert "NORMAL" in pulse

        why = render_why_alert_panel(
            primary_sensor="pump_flow",
            primary_sigma=3.8,
            supporting_sensors=[("pump_pressure", 2.5)],
            correlated_info=[{"sensor_id": "pump_speed", "correlation": 0.95}],
            trend="Increasing",
            model_agreement_dict={"if": True, "ae": True},
            narrative="Process dev observed",
        )
        assert "WHY DID THE SYSTEM ALERT?" in why
        assert "pump_flow" in why

