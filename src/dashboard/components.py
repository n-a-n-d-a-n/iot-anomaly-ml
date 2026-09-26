"""Reusable Plotly chart components for the Industrial IoT Anomaly Intelligence UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dark theme palette defaults
DARK_BG = "#111827"
DARK_PLOT_BG = "#0B1120"
BORDER_COLOR = "#1F2937"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"


def create_anomaly_timeline(
    df: pd.DataFrame,
    sensor_col: str = "sensor_id",
    value_col: str = "value",
    time_col: str = "timestamp",
    anomaly_col: str = "is_anomaly",
) -> go.Figure:
    """Create a timeline chart with anomaly markers overlaid.

    Args:
        df: DataFrame with sensor readings.
        sensor_col: Column identifying the sensor.
        value_col: Column with measured values.
        time_col: Column with timestamps.
        anomaly_col: Binary column flagging anomalies.

    Returns:
        Plotly figure.
    """
    fig = px.line(df, x=time_col, y=value_col, color=sensor_col)

    anomalies = df[df[anomaly_col] == 1]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies[time_col],
                y=anomalies[value_col],
                mode="markers",
                marker={"color": "#ef4444", "size": 10, "symbol": "x"},
                name="Anomalies",
                showlegend=True,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
    )
    return fig


def create_contribution_chart(
    contributions: dict[str, float],
    title: str = "Feature Contributions",
) -> go.Figure:
    """Create a bar chart for feature contribution scores.

    Args:
        contributions: Mapping of feature name to contribution score.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    sorted_items = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    fig = go.Figure(
        go.Bar(
            x=[item[0] for item in sorted_items],
            y=[item[1] for item in sorted_items],
            marker_color=[
                px.colors.sequential.Reds[min(int(item[1] * 9), 8)] for item in sorted_items
            ],
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Sensor",
        yaxis_title="Contribution Score",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
    )
    return fig


def create_health_gauge(anomaly_rate: float, sensor_name: str) -> go.Figure:
    """Create a gauge chart for sensor health status.

    Args:
        anomaly_rate: Anomaly rate in [0, 1].
        sensor_name: Sensor identifier used as chart title.

    Returns:
        Plotly gauge figure.
    """
    color = "#10b981" if anomaly_rate < 0.05 else "#f59e0b" if anomaly_rate < 0.1 else "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=anomaly_rate * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": sensor_name, "font": {"color": TEXT_COLOR, "size": 13}},
            number={"suffix": "%", "font": {"color": TEXT_COLOR, "size": 20}},
            gauge={
                "axis": {"range": [0, 20], "tickcolor": MUTED_TEXT},
                "bar": {"color": color},
                "bgcolor": "#1f2937",
                "steps": [
                    {"range": [0, 5], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [5, 10], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [10, 20], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.75,
                    "value": 10,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT_COLOR},
        margin={"t": 35, "b": 10, "l": 25, "r": 25},
        height=180,
    )
    return fig


def create_hero_gauge(
    score: float,
    threshold: float = 0.6061,
    title: str = "CURRENT ANOMALY SCORE",
) -> go.Figure:
    """Create a dominant hero gauge indicator for real-time plant anomaly status.

    Args:
        score: Ensemble anomaly score in [0, 1].
        threshold: Calibrated ensemble threshold.
        title: Title string.

    Returns:
        Plotly gauge figure.
    """
    is_anom = score >= threshold
    bar_color = "#ef4444" if is_anom else ("#f59e0b" if score >= threshold * 0.8 else "#10b981")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"color": MUTED_TEXT, "size": 12, "family": "Inter"}},
            number={
                "font": {"color": TEXT_COLOR, "size": 34, "family": "JetBrains Mono"},
                "valueformat": ".4f",
            },
            gauge={
                "axis": {"range": [0, 1.0], "tickcolor": MUTED_TEXT, "tickwidth": 1, "nticks": 6},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#1f2937",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 0.8], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [threshold * 0.8, threshold], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [threshold, 1.0], "color": "rgba(239, 68, 68, 0.28)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT_COLOR},
        margin={"t": 35, "b": 15, "l": 20, "r": 20},
        height=220,
    )
    return fig


def create_score_timeline(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    threshold: float = 0.6061,
    time_col: str = "timestamp",
    anomaly_col: str = "is_anomaly",
    title: str = "ANOMALY ACTIVITY • ENSEMBLE SCORE VS CALIBRATED THRESHOLD",
) -> go.Figure:
    """Interactive dark anomaly timeline with score vs threshold and anomaly highlights.

    Args:
        df: DataFrame with timestamps, scores, and anomaly labels.
        score_col: Column with continuous anomaly scores.
        threshold: Decision threshold.
        time_col: Timestamp column.
        anomaly_col: Binary indicator column.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    # Resilient score column resolution
    if score_col not in df.columns:
        for candidate in ["anomaly_score", "score", "ensemble_score"]:
            if candidate in df.columns:
                score_col = candidate
                break

    fig = go.Figure()

    # Continuous ensemble anomaly score line
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[score_col],
            mode="lines",
            line={"color": "#38bdf8", "width": 1.8},
            name="Ensemble Score",
        )
    )

    # Decision threshold line
    fig.add_trace(
        go.Scatter(
            x=[df[time_col].min(), df[time_col].max()],
            y=[threshold, threshold],
            mode="lines",
            line={"color": "#ef4444", "width": 2, "dash": "dash"},
            name=f"Threshold ({threshold:.4f})",
        )
    )

    # Detected anomalies
    if anomaly_col in df.columns:
        anom_df = df[df[anomaly_col] == 1]
        if not anom_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_df[time_col],
                    y=anom_df[score_col],
                    mode="markers",
                    marker={"color": "#ef4444", "size": 7, "symbol": "circle", "line": {"color": "#ffffff", "width": 1}},
                    name="Detected Anomalies",
                )
            )

    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": TEXT_COLOR, "family": "Inter"}},
        xaxis_title="Timeline",
        yaxis_title="Score",
        yaxis={"range": [-0.02, 1.05], "gridcolor": "#1f2937"},
        xaxis={"gridcolor": "#1f2937"},
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 45, "b": 35, "l": 45, "r": 20},
        height=340,
    )
    return fig


def create_sensor_timeseries_view(
    df: pd.DataFrame,
    sensor_col: str,
    time_col: str = "timestamp",
    anomaly_col: str = "is_anomaly",
    ref_mean: float | None = None,
    ref_std: float | None = None,
    highlight_time: pd.Timestamp | datetime | None = None,
    title: str | None = None,
) -> go.Figure:
    """Time-series chart for a single sensor with normal reference envelope.

    Args:
        df: DataFrame containing sensor observations.
        sensor_col: Column name of the sensor to plot.
        time_col: Timestamp column.
        anomaly_col: Anomaly flag column.
        ref_mean: Baseline mean for normal operation.
        ref_std: Baseline standard deviation for normal operation.
        highlight_time: Timestamp of a currently selected anomaly to highlight.
        title: Optional plot title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    # Normal reference band (mean +/- 2 std)
    if ref_mean is not None and ref_std is not None and ref_std > 0:
        upper = ref_mean + 2 * ref_std
        lower = ref_mean - 2 * ref_std
        times = df[time_col]
        fig.add_trace(
            go.Scatter(
                x=list(times) + list(times[::-1]),
                y=[upper] * len(times) + [lower] * len(times),
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.12)",
                line={"color": "rgba(255,255,255,0)"},
                name="Normal Operating Envelope (±2σ)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[times.min(), times.max()],
                y=[ref_mean, ref_mean],
                mode="lines",
                line={"color": "#10b981", "width": 1.2, "dash": "dot"},
                name="Normal Reference Mean",
            )
        )

    # Observed sensor values
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[sensor_col],
            mode="lines",
            line={"color": "#38bdf8", "width": 1.8},
            name="Sensor Reading",
        )
    )

    # Highlight anomalous points
    if anomaly_col in df.columns:
        anom_points = df[df[anomaly_col] == 1]
        if not anom_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_points[time_col],
                    y=anom_points[sensor_col],
                    mode="markers",
                    marker={"color": "#ef4444", "size": 8, "symbol": "diamond"},
                    name="Anomaly Event",
                )
            )

    # Selected event vertical marker
    if highlight_time is not None:
        fig.add_vline(
            x=highlight_time,
            line_width=2,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Selected Event",
            annotation_position="top right",
            annotation_font_color="#f59e0b",
        )

    fig.update_layout(
        title={"text": title or f"SENSOR TELEMETRY: {sensor_col.upper()}", "font": {"size": 13, "color": TEXT_COLOR}},
        xaxis_title="Timestamp",
        yaxis_title="Measured Value",
        xaxis={"gridcolor": "#1f2937"},
        yaxis={"gridcolor": "#1f2937"},
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 45, "b": 35, "l": 45, "r": 20},
        height=320,
    )
    return fig


def create_model_agreement_chart(
    model_scores: dict[str, float],
    threshold: float = 0.6061,
) -> go.Figure:
    """Horizontal bar chart showing individual detector scores against calibrated threshold.

    Args:
        model_scores: Mapping from detector name to normalized anomaly score in [0, 1].
        threshold: Ensemble decision threshold.

    Returns:
        Plotly figure.
    """
    labels = list(model_scores.keys())
    scores = [model_scores[k] for k in labels]
    display_names = {
        "isolation_forest": "Isolation Forest",
        "autoencoder": "Autoencoder",
        "dbscan": "DBSCAN",
        "gmm": "GMM",
        "ensemble": "Ensemble",
    }
    clean_labels = [display_names.get(lbl, lbl.title()) for lbl in labels]
    colors = ["#ef4444" if s >= threshold else "#10b981" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=clean_labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{s:.3f} ({'ANOMALY' if s >= threshold else 'NORMAL'})" for s in scores],
            textposition="auto",
            textfont={"family": "JetBrains Mono", "color": "#ffffff"},
        )
    )

    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Threshold ({threshold:.4f})",
        annotation_position="top right",
        annotation_font_color="#ef4444",
    )

    fig.update_layout(
        title={"text": "MODEL DETECTOR AGREEMENT", "font": {"size": 13, "color": TEXT_COLOR}},
        xaxis_title="Normalized Score",
        yaxis_title="Model",
        xaxis={"range": [0, 1.05], "gridcolor": "#1f2937"},
        yaxis={"gridcolor": "#1f2937"},
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
        height=260,
        margin={"t": 35, "b": 35, "l": 110, "r": 20},
    )
    return fig


def create_correlation_bar(
    correlations: list[dict[str, Any]],
    target_sensor: str,
) -> go.Figure:
    """Bar chart displaying strongly correlated sensors and Pearson r values.

    Args:
        correlations: List of dicts with 'sensor_id' and 'correlation'.
        target_sensor: Main anomalous sensor name.

    Returns:
        Plotly figure.
    """
    if not correlations:
        fig = go.Figure()
        fig.update_layout(
            title="No strongly correlated variables (|r| < 0.40)",
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_PLOT_BG,
            font={"color": TEXT_COLOR},
            height=260,
        )
        return fig

    sensors = [c["sensor_id"] for c in correlations]
    r_vals = [c["correlation"] for c in correlations]
    colors = ["#38bdf8" if r >= 0 else "#a855f7" for r in r_vals]

    fig = go.Figure(
        go.Bar(
            x=sensors,
            y=r_vals,
            marker={"color": colors},
            text=[f"r = {r:+.4f}" for r in r_vals],
            textposition="auto",
            textfont={"family": "JetBrains Mono", "color": "#ffffff"},
        )
    )
    fig.update_layout(
        title={"text": f"VARIABLES CORRELATED WITH {target_sensor.upper()}", "font": {"size": 13, "color": TEXT_COLOR}},
        xaxis_title="Correlated Sensor",
        yaxis_title="Pearson r",
        yaxis={"range": [-1.05, 1.05], "gridcolor": "#1f2937"},
        xaxis={"gridcolor": "#1f2937"},
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
        height=260,
        margin={"t": 35, "b": 35, "l": 45, "r": 20},
    )
    return fig


def create_radar_comparison_chart(eval_results: dict[str, dict[str, float]]) -> go.Figure:
    """Interactive radar/spider chart comparing all 4 detectors and the ensemble.

    Args:
        eval_results: Nested dictionary of metrics per model.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    categories = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    color_map = {
        "isolation_forest": "#38bdf8",
        "autoencoder": "#a855f7",
        "dbscan": "#f59e0b",
        "gmm": "#10b981",
        "ensemble": "#ef4444",
    }
    names_map = {
        "isolation_forest": "Isolation Forest",
        "autoencoder": "Autoencoder",
        "dbscan": "DBSCAN",
        "gmm": "GMM",
        "ensemble": "Weighted Ensemble",
    }

    for model_key, res in eval_results.items():
        vals = [
            res.get("precision", 0.0),
            res.get("recall", 0.0),
            res.get("f1", 0.0),
            res.get("auc_roc", 0.0),
            res.get("pr_auc", 0.0),
        ]
        # Close the loop on polar chart
        vals.append(vals[0])
        color = color_map.get(model_key, "#ffffff")
        name = names_map.get(model_key, model_key.title())
        is_ens = model_key == "ensemble"

        trace_kwargs: dict[str, Any] = {
            "r": vals,
            "theta": categories + [categories[0]],
            "name": name,
            "line": {"color": color, "width": 3 if is_ens else 1.5},
        }
        if is_ens:
            trace_kwargs["fill"] = "toself"
            trace_kwargs["fillcolor"] = "rgba(239, 68, 68, 0.15)"

        fig.add_trace(go.Scatterpolar(**trace_kwargs))

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1.0], "gridcolor": "#1f2937", "tickcolor": MUTED_TEXT},
            "angularaxis": {"gridcolor": "#1f2937"},
            "bgcolor": DARK_PLOT_BG,
        },
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        font={"color": TEXT_COLOR},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
        title={"text": "MULTIVARIATE BENCHMARK RADAR COMPARISON", "font": {"size": 13, "color": TEXT_COLOR}},
        height=380,
        margin={"t": 40, "b": 60, "l": 40, "r": 40},
    )
    return fig


def create_correlation_heatmap(corr_df: pd.DataFrame, title: str = "PROCESS CORRELATION MATRIX") -> go.Figure:
    """Create correlation heatmap for process variables.

    Args:
        corr_df: Pairwise correlation DataFrame.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "Pearson r", "tickfont": {"color": TEXT_COLOR}},
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": TEXT_COLOR}},
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
        xaxis={"tickangle": -45, "tickfont": {"size": 9}},
        yaxis={"tickfont": {"size": 9}},
        height=540,
        margin={"t": 45, "b": 70, "l": 90, "r": 30},
    )
    return fig


def create_top_contributors_chart(
    contributors: list[Any],
    max_bars: int = 5,
    title: str = "TOP CONTRIBUTING SENSORS (NORMALIZED DEVIATION)",
) -> go.Figure:
    """Create an aligned Plotly horizontal bar chart for feature contributions."""
    if not contributors:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "No contributing sensor deviations recorded", "font": {"size": 13, "color": TEXT_COLOR}},
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_PLOT_BG,
            font={"color": TEXT_COLOR},
            height=180,
        )
        return fig

    items = contributors[:max_bars]
    # Reverse order so rank 01 is at the top of the chart
    items = items[::-1]

    labels = []
    devs = []
    colors = []
    text_labels = []

    for item in items:
        if hasattr(item, "sensor_id"):
            rank = getattr(item, "rank", 1)
            s_id = item.sensor_id
            dev = float(getattr(item, "normalized_deviation", 0.0))
        elif isinstance(item, (tuple, list)):
            rank = item[0] if len(item) > 2 else 1
            s_id = item[1] if len(item) > 2 else item[0]
            dev = float(item[2] if len(item) > 2 else item[1])
        else:
            rank = 1
            s_id = str(item)
            dev = 0.0

        label = f"{rank:02d}  {s_id}"
        labels.append(label)
        devs.append(dev)
        if abs(dev) >= 3.0:
            colors.append("#F43F5E")
        elif abs(dev) >= 2.0:
            colors.append("#F59E0B")
        else:
            colors.append("#38BDF8")
        text_labels.append(f"{dev:+.2f}σ")

    fig = go.Figure(
        go.Bar(
            x=devs,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=text_labels,
            textposition="auto",
            textfont={"family": "monospace", "color": "#FFFFFF", "size": 11},
        )
    )

    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": TEXT_COLOR}},
        xaxis_title="Normalized Deviation (σ)",
        yaxis_title="",
        xaxis={"gridcolor": BORDER_COLOR, "zerolinecolor": "#475569"},
        yaxis={"gridcolor": BORDER_COLOR},
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT_BG,
        font={"color": TEXT_COLOR},
        height=max(180, len(items) * 44 + 60),
        margin={"t": 35, "b": 35, "l": 160, "r": 30},
    )
    return fig
