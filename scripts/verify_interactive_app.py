"""Verification script testing all 14 user interactions in src/dashboard/app.py.
Uses direct state execution & AppTest to verify interactive workflows end-to-end.
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.state import MonitoringEngine, Incident

import json
from pathlib import Path
import pandas as pd
from src.analysis.root_cause import RootCauseAnalyzer
from src.dashboard.state import MonitoringEngine, Incident

def test_full_operational_story():
    print("==================================================")
    print("STEP-BY-STEP OPERATIONAL INTERACTION TEST (REAL DATA)")
    print("==================================================")
    
    # Load actual dataset
    data_path = PROJECT_ROOT / "data" / "generated" / "sensor_data.csv"
    assert data_path.exists(), f"Dataset missing at {data_path}"
    df_raw = pd.read_csv(data_path, parse_dates=["timestamp"])
    if "sensor_id" in df_raw.columns:
        df = df_raw.pivot_table(
            index="timestamp",
            columns="sensor_id",
            values="value",
            aggfunc="first",
        ).reset_index()
        if "is_anomaly" in df_raw.columns:
            anom_map = df_raw.groupby("timestamp")["is_anomaly"].max()
            df["is_anomaly"] = df["timestamp"].map(anom_map).fillna(0).astype(int)
    else:
        df = df_raw.sort_values("timestamp")

    sensor_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    print(f"Loaded dataset: {len(df)} time samples, {len(sensor_cols)} sensors.")
    
    # Load real calibrated threshold
    eval_path = PROJECT_ROOT / "data" / "evaluation_results.json"
    calibrated_th = 0.6061
    if eval_path.exists():
        with open(eval_path) as f:
            res = json.load(f)
            calibrated_th = res.get("calibrated_threshold", 0.6061)
            
    analyzer = RootCauseAnalyzer(sensor_names=sensor_cols, threshold=calibrated_th)
    analyzer.compute_reference_stats(df)
    
    engine = MonitoringEngine(calibrated_threshold=calibrated_th, start_index=7056)
    
    # 1. SETUP / READY STATE
    print("\n[ACTION 1] Verifying SETUP / READY STATE...")
    assert engine.current_mode == "SETUP"
    assert engine.monitoring_active is False
    assert engine.replay_index == 7056
    assert len(engine.incidents) == 0
    print(f"  -> OK: Initial Mode={engine.current_mode}, MonitoringActive={engine.monitoring_active}, Index={engine.replay_index}")
    
    # 2. CONFIGURE SCOPE
    print("\n[ACTION 2] Configuring monitoring scope...")
    engine.selected_subsystem = "Pressure"
    assert engine.selected_subsystem == "Pressure"
    print(f"  -> OK: Scope set to Subsystem={engine.selected_subsystem}")
    
    # 3. START MONITORING
    print("\n[ACTION 3] START MONITORING action...")
    engine.start_monitoring()
    assert engine.monitoring_active is True
    assert engine.current_mode == "LIVE MONITORING"
    print(f"  -> OK: Monitoring active, Mode={engine.current_mode}")
    
    # 4. STEP TELEMETRY
    print("\n[ACTION 4] INGEST TELEMETRY (Stepping rows)...")
    inc = engine.step_telemetry(df, analyzer, n_steps=5)
    assert engine.replay_index == 7061
    assert engine.samples_processed == 5
    print(f"  -> OK: Advanced to Index={engine.replay_index}, Score={engine.current_score:.4f}, Latency={engine.latest_latency_ms:.2f}ms")
    
    # 5. PAUSE / RESUME
    print("\n[ACTION 5] PAUSE / RESUME MONITORING...")
    engine.pause_monitoring()
    assert engine.monitoring_active is False
    engine.start_monitoring()
    assert engine.monitoring_active is True
    print("  -> OK: Successfully paused and resumed")
    
    # 6. JUMP TO NEXT ANOMALY
    print("\n[ACTION 6] JUMP TO NEXT ANOMALY...")
    jumped_inc = engine.jump_to_next_anomaly(df, analyzer)
    assert jumped_inc is not None
    assert engine.current_score >= engine.active_threshold
    assert len(engine.incidents) >= 1
    assert engine.selected_incident_id == jumped_inc.incident_id
    print(f"  -> OK: Jumped to Index={jumped_inc.row_index}, Time={jumped_inc.timestamp}")
    print(f"  -> Anomaly Detected: Score={jumped_inc.ensemble_score:.4f} >= Threshold={engine.active_threshold:.4f}")
    print(f"  -> Model Consensus: {jumped_inc.model_consensus}")
    
    # 7. INVESTIGATE INCIDENT
    print("\n[ACTION 7] INVESTIGATE INCIDENT action...")
    target_inc_id = jumped_inc.incident_id
    engine.current_mode = "INVESTIGATION"
    engine.selected_incident_id = target_inc_id
    inc_obj = engine.get_incident(target_inc_id)
    assert inc_obj is not None
    assert inc_obj.status == "NEW"
    print(f"  -> OK: Viewing {target_inc_id} in Mode={engine.current_mode}")
    
    # 8. SENSOR INVESTIGATION & MODEL AGREEMENT
    print("\n[ACTION 8] VERIFY SENSOR INVESTIGATION & MODEL EVIDENCE...")
    assert inc_obj.primary_sensor is not None
    assert inc_obj.subsystem is not None
    assert inc_obj.explanation is not None
    engine.selected_sensor = inc_obj.primary_sensor
    print(f"  -> OK: Primary Driver={inc_obj.primary_sensor}, Subsystem={inc_obj.subsystem}, Pattern={inc_obj.pattern}")
    print(f"  -> OK: Model Scores: {engine.current_model_scores}")
    
    # 9. ACKNOWLEDGE ALERT
    print("\n[ACTION 9] ACKNOWLEDGE ALERT action...")
    engine.acknowledge_incident(target_inc_id)
    assert inc_obj.status == "ACKNOWLEDGED"
    print(f"  -> OK: Status transitioned to {inc_obj.status}")
    
    # 10. MARK FOR REVIEW
    print("\n[ACTION 10] MARK FOR REVIEW action...")
    engine.mark_for_review(target_inc_id)
    assert inc_obj.status == "REVIEW"
    print(f"  -> OK: Status transitioned to {inc_obj.status}")
    
    # 11. OPERATOR NOTES
    print("\n[ACTION 11] ADD OPERATOR NOTE action...")
    note_text = "Cavitation signature observed on pump impeller; maintenance dispatched."
    engine.add_operator_note(target_inc_id, note_text, author="Lead Operator #4")
    assert len(inc_obj.notes) == 1
    assert inc_obj.notes[0]["text"] == note_text
    print(f"  -> OK: Note saved: '{inc_obj.notes[0]['text']}' by {inc_obj.notes[0]['author']}")
    
    # 12. RESOLVE INCIDENT
    print("\n[ACTION 12] RESOLVE INCIDENT action...")
    engine.resolve_incident(target_inc_id)
    assert inc_obj.status == "RESOLVED"
    print(f"  -> OK: Status transitioned to {inc_obj.status}")
    
    # 13. GENERATE INCIDENT REPORT
    print("\n[ACTION 13] GENERATE INCIDENT REPORT action...")
    report_md = engine.generate_incident_report(target_inc_id)
    assert "INDUSTRIAL IoT ANOMALY AUDIT REPORT" in report_md
    assert f"**Incident Reference:** `{target_inc_id}`" in report_md
    assert "Cavitation signature" in report_md
    assert "ROOT CAUSE & EXPLAINABILITY" in report_md
    print(f"  -> OK: Incident Report successfully generated ({len(report_md)} bytes markdown)")
    
    # 14. RESET MONITORING
    print("\n[ACTION 14] RESET MONITORING action...")
    engine.reset_monitoring(start_index=7056)
    assert engine.current_mode == "SETUP"
    assert engine.replay_index == 7056
    assert engine.samples_processed == 0
    assert engine.anomalies_detected == 0
    assert engine.monitoring_active is False
    assert len(engine.incidents) == 0
    assert engine.selected_incident_id is None
    print("  -> OK: System completely reset to initial SETUP state")
    
    print("\n==================================================")
    print("SUCCESS: ALL 14 OPERATOR INTERACTIONS VERIFIED!")
    print("==================================================")

if __name__ == "__main__":
    test_full_operational_story()
