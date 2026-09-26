"""Dark Industrial SCADA / Mission Control Design System & CSS Theme."""

from __future__ import annotations

import textwrap
from typing import Any

# Semantic industrial color definitions
COLOR_BG_DARK = "#0B1120"        # Deep obsidian slate
COLOR_PANEL_BG = "#111827"       # Surface navy
COLOR_PANEL_ELEVATED = "#1E293B" # Elevated card navy
COLOR_PANEL_BORDER = "#1F2937"   # Structural border
COLOR_TEXT_PRIMARY = "#F8FAFC"   # High-contrast white
COLOR_TEXT_SECONDARY = "#94A3B8" # Muted slate gray
COLOR_BLUE = "#38BDF8"           # Telemetry cyan-blue
COLOR_NORMAL = "#10B981"         # Emerald green
COLOR_WARNING = "#F59E0B"        # Amber warning
COLOR_ANOMALY = "#F43F5E"        # Rose red critical
COLOR_PURPLE = "#A78BFA"         # Model purple accent

# Deprecated aliases preserved for backward compatibility
COLOR_INFO = COLOR_BLUE
COLOR_ACCENT = COLOR_BLUE
COLOR_TEXT_MUTED = COLOR_TEXT_SECONDARY

INDUSTRIAL_CSS = textwrap.dedent("""
<style>
/* Reset and global dark control-room theme */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0B1120 !important;
    color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    box-sizing: border-box !important;
}

/* Eliminate Streamlit default white header */
[data-testid="stHeader"], header[data-testid="stHeader"], .stAppHeader {
    background-color: #0B1120 !important;
    border-bottom: 1px solid #1F2937 !important;
    color: #F8FAFC !important;
}

[data-testid="stToolbar"] {
    color: #94A3B8 !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Sidebar styling: width 270px, high-contrast text, clear active state */
[data-testid="stSidebar"] {
    background-color: #0B1120 !important;
    border-right: 1px solid #1F2937 !important;
    width: 270px !important;
    min-width: 250px !important;
    max-width: 280px !important;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: #F8FAFC !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    border-radius: 4px !important;
    display: flex !important;
    align-items: center !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(56, 189, 248, 0.08) !important;
    color: #38BDF8 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(56, 189, 248, 0.15) !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    color: #38BDF8 !important;
    font-weight: 600 !important;
}

/* Global button styling */
.stButton > button {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background-color: #38BDF8 !important;
    color: #0B1120 !important;
    border-color: #38BDF8 !important;
}

/* Standardized design system classes */
.app-shell {
    width: 100%;
    box-sizing: border-box;
    padding: 0;
    margin: 0;
}

.page-header {
    margin-bottom: 20px;
}

.section-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #94A3B8;
    text-transform: uppercase;
    margin: 16px 0 8px 0;
}

.metric-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 12px 14px;
    box-sizing: border-box;
    width: 100%;
}

.status-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 14px 18px;
    box-sizing: border-box;
    width: 100%;
}

.pulse-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 10px 8px;
    text-align: center;
    box-sizing: border-box;
    width: 100%;
    min-height: 82px;
}

.diagnostic-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-left: 4px solid #F43F5E;
    border-radius: 6px;
    padding: 16px;
    box-sizing: border-box;
    width: 100%;
    margin: 12px 0;
}

.alert-banner {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid #F43F5E;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    box-sizing: border-box;
    width: 100%;
}

.model-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 14px;
    box-sizing: border-box;
    width: 100%;
}

.sensor-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 14px;
    box-sizing: border-box;
    width: 100%;
}

.system-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 14px;
    box-sizing: border-box;
    width: 100%;
}

/* Custom header bar */
.ind-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 12px 18px;
    margin-bottom: 16px;
    box-sizing: border-box;
    width: 100%;
}

.ind-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.ind-logo-badge {
    background: #0284C7;
    color: #FFFFFF;
    font-family: monospace;
    font-weight: 700;
    font-size: 13px;
    padding: 5px 9px;
    border-radius: 4px;
    letter-spacing: 1px;
}

.ind-header-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #F8FAFC;
    margin: 0;
    line-height: 1.2;
}

.ind-header-subtitle {
    font-size: 12px;
    color: #94A3B8;
    margin: 0;
    letter-spacing: 0.3px;
}

.ind-header-right {
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: monospace;
    font-size: 11px;
}

.ind-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10B981;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: 600;
}

.ind-pill-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #10B981;
}

.ind-badge-sim {
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38BDF8;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: 600;
}

/* Plant pulse card internal styles */
.pulse-name {
    font-size: 11px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.pulse-status {
    font-family: monospace;
    font-size: 12px;
    font-weight: 700;
}

.pulse-status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 4px;
}

.pulse-count {
    font-size: 10px;
    color: #64748B;
    margin-top: 4px;
}

/* Model agreement pills */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 11px;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}

.model-pill-anom {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #F43F5E;
}

.model-pill-norm {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10B981;
}

/* Plotly chart container dark styling */
.js-plotly-plot {
    border: 1px solid #1F2937 !important;
    border-radius: 6px !important;
    background-color: #111827 !important;
    box-sizing: border-box !important;
}

/* Dataframe dark table styling */
[data-testid="stDataFrame"] {
    border: 1px solid #1F2937 !important;
    border-radius: 6px !important;
    background-color: #111827 !important;
}
</style>
""").strip()


def render_app_header(*args: Any, **kwargs: Any) -> str:
    """Render top application shell header with zero leading indentation."""
    mode = kwargs.get("mode") or kwargs.get("mode_or_updated")
    last_updated = kwargs.get("last_updated")

    if args:
        if len(args) == 1:
            val = str(args[0])
            if any(char in val for char in ["2024-", "2025-", "2026-", ":"]) and not val.startswith("MODE:"):
                last_updated = last_updated or val
                mode = mode or "LIVE MONITORING"
            else:
                mode = mode or val
        elif len(args) >= 2:
            mode = mode or str(args[0])
            last_updated = last_updated or str(args[1])

    if not mode:
        mode = "LIVE MONITORING"
    if not last_updated:
        last_updated = "LIVE TELEMETRY"

    html = f"""<div class="ind-header">
<div class="ind-header-left">
<span class="ind-logo-badge">AGRI-IIoT</span>
<div>
<h1 class="ind-header-title">INDUSTRIAL IoT ANOMALY INTELLIGENCE</h1>
<p class="ind-header-subtitle">Multivariate Unsupervised Monitoring Platform • Continuous Process Safety</p>
</div>
</div>
<div class="ind-header-right">
<span class="ind-status-pill"><span class="ind-pill-dot"></span> SYSTEM ONLINE</span>
<span class="ind-badge-sim">MODE: {mode}</span>
<span style="color: #94A3B8;">MODELS: <strong style="color: #F8FAFC;">4 ACTIVE</strong></span>
<span style="color: #94A3B8;">SENSORS: <strong style="color: #F8FAFC;">28 MONITORED</strong></span>
<span style="color: #64748B;">{last_updated}</span>
</div>
</div>"""
    return html


def render_pulse_card(name: str, status: str = "NORMAL", count: int = 4) -> str:
    """Render a single subsystem pulse card with zero indentation."""
    dot_color = (
        COLOR_ANOMALY if status == "ANOMALY" else (COLOR_WARNING if status == "WARNING" else COLOR_NORMAL)
    )
    html = f"""<div class="pulse-card">
<div class="pulse-name">{name}</div>
<div class="pulse-status" style="color: {dot_color};">
<span class="pulse-status-dot" style="background-color: {dot_color};"></span>{status}
</div>
<div class="pulse-count">{count} sensors</div>
</div>"""
    return html


def render_plant_pulse(subsystems_status: list[dict[str, Any]]) -> str:
    """Render the 7-subsystem pulse grid with zero indentation."""
    cards_html = []
    for sub in subsystems_status:
        name = sub["name"]
        status = sub.get("status", "NORMAL")
        count = sub.get("count", 4)
        cards_html.append(render_pulse_card(name, status, count))
    
    grid_inner = "".join(cards_html)
    return f"""<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; margin-bottom: 16px; width: 100%; box-sizing: border-box;">{grid_inner}</div>"""


def render_why_alert_panel(
    primary_sensor: str,
    primary_sigma: float,
    supporting_sensors: list[tuple[str, float]],
    correlated_info: list[dict[str, Any]],
    trend: str,
    model_agreement_dict: dict[str, bool],
    narrative: str,
) -> str:
    """Render the signature 'WHY DID THE SYSTEM ALERT?' panel with conservative language and zero indentation."""
    model_badges = []
    for m_name, is_anom in model_agreement_dict.items():
        disp = m_name.replace("_", " ").upper()
        if is_anom:
            model_badges.append(f'<span class="model-pill model-pill-anom">● {disp}</span>')
        else:
            model_badges.append(f'<span class="model-pill model-pill-norm">○ {disp}</span>')
    badges_str = "".join(model_badges)

    sup_str = ", ".join([f"<strong>{s}</strong> ({dev:+.2f}σ)" for s, dev in supporting_sensors]) if supporting_sensors else "None above threshold"
    corr_str = ", ".join([f"<strong>{c['sensor_id']}</strong> (r = {c['correlation']:+.2f})" for c in correlated_info]) if correlated_info else "No strong process correlations observed"

    html = f"""<div class="diagnostic-card">
<div style="font-size: 13px; font-weight: 700; color: #F43F5E; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
<span>⚠️</span> WHY DID THE SYSTEM ALERT?
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; width: 100%; box-sizing: border-box;">
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-bottom: 3px;">Primary Driver Signal</div>
<div style="font-family: monospace; font-size: 15px; font-weight: 700; color: #F43F5E;">
{primary_sensor} <span style="font-size: 13px; color: #FDA4AF;">({primary_sigma:+.2f}σ deviation)</span>
</div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-top: 10px; margin-bottom: 3px;">Supporting Deviations</div>
<div style="font-size: 12px; color: #E2E8F0;">{sup_str}</div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-top: 10px; margin-bottom: 3px;">Correlated Process Coupling</div>
<div style="font-size: 12px; color: #94A3B8;">{corr_str}</div>
</div>
<div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-bottom: 3px;">Observed Temporal Trend</div>
<div style="font-family: monospace; font-size: 13px; color: #F59E0B;">{trend}</div>
<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-top: 10px; margin-bottom: 3px;">Multi-Model Detector Agreement</div>
<div style="margin-top: 4px;">{badges_str}</div>
</div>
</div>
<div style="background: #0B1120; border: 1px solid #1F2937; border-radius: 4px; padding: 10px 14px; font-size: 13px; color: #E2E8F0; line-height: 1.45; margin-top: 12px;">
<strong style="color: #94A3B8;">System Interpretation:</strong> {narrative}
</div>
</div>"""
    return html
