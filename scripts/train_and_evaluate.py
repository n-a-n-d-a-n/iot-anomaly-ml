"""Train all anomaly detection models and save evaluation results.

Usage:
    python -m scripts.train_and_evaluate [--data-path DATA_CSV] [--days DAYS] [--epochs EPOCHS] [--force-generate]

Generates ``data/evaluation_results.json`` and ``data/evaluation_results.csv``
consumed by the dashboard's Model Comparison tab and benchmark evaluations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generator import IoTDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.autoencoder import AutoencoderDetector
from src.models.dbscan_detector import DBSCANDetector
from src.models.ensemble import EnsembleDetector
from src.models.evaluator import EvaluationResult, ModelEvaluator
from src.models.gmm_detector import GMMDetector
from src.models.isolation_forest import IsolationForestDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = Path("data")
GENERATED_CSV = DATA_DIR / "generated" / "sensor_data.csv"
EVAL_JSON = DATA_DIR / "evaluation_results.json"
EVAL_CSV = DATA_DIR / "evaluation_results.csv"


def load_or_generate_data(
    data_path: Path, days: int = 7, force_generate: bool = False
) -> pd.DataFrame:
    """Load existing sensor data or generate a new correlated Industrial IoT dataset."""
    if data_path.exists() and not force_generate:
        logger.info("Loading existing data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        if df["sensor_id"].nunique() == 28:
            return df
        logger.info(
            "Existing CSV has %d sensor features (expected 28) — regenerating fresh dataset",
            df["sensor_id"].nunique(),
        )

    logger.info("Generating %d days of synthetic Industrial IoT process data (28 features)...", days)
    gen = IoTDataGenerator(seed=42)
    df = gen.generate(days=days)
    gen.save_to_csv(df, str(data_path))
    return df


def main(
    data_path: Path = GENERATED_CSV,
    days: int = 7,
    epochs: int = 30,
    force_generate: bool = False,
) -> None:
    t_start_pipeline = time.perf_counter()

    t_start_gen = time.perf_counter()
    df = load_or_generate_data(data_path, days=days, force_generate=force_generate)
    t_gen = time.perf_counter() - t_start_gen

    # Dataset validation & statistics
    gen = IoTDataGenerator(seed=42)
    ds_stats = gen.validate_dataset(df)
    correlations = gen.compute_correlations(df)

    logger.info("=== INDUSTRIAL IoT DATASET VALIDATION ===")
    for k, v in ds_stats.items():
        logger.info("  %s: %s", k, v)

    logger.info("=== PHYSICAL PROCESS CORRELATIONS ===")
    for k, v in correlations.items():
        logger.info("  %s: %.4f", k, v)

    # Preprocess
    t_start_prep = time.perf_counter()
    preprocessor = DataPreprocessor(scaler_type="standard")
    pivoted = preprocessor.pivot_sensors(df)
    X_scaled, y = preprocessor.fit_transform(pivoted)
    feature_names = preprocessor.feature_names
    t_prep = time.perf_counter() - t_start_prep

    if y is None:
        raise RuntimeError("Dataset has no is_anomaly labels — cannot evaluate")

    # Temporal Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(
        X_scaled, y, train_ratio=0.70, val_ratio=0.15
    )

    logger.info(
        "Split sizes — Train: %d (anomalies: %d), Val: %d (anomalies: %d), Test: %d (anomalies: %d)",
        len(X_train),
        int(y_train.sum()),
        len(X_val),
        int(y_val.sum()),
        len(X_test),
        int(y_test.sum()),
    )

    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation split must contain both normal and anomalous samples.")
    if len(np.unique(y_test)) < 2:
        raise ValueError("Test split must contain both normal and anomalous samples for ROC-AUC evaluation.")

    results: dict[str, dict] = {}

    # --- 1. Isolation Forest ---
    logger.info("Training Isolation Forest...")
    t0 = time.perf_counter()
    iso = IsolationForestDetector(contamination=0.05, random_state=42)
    iso.fit(X_train, feature_names)
    t_iso = time.perf_counter() - t0
    iso_pred = iso.predict(X_test)
    iso_scores = iso.score_samples(X_test)
    iso_eval = ModelEvaluator.evaluate(y_test, iso_pred, iso_scores)
    results["Isolation Forest"] = iso_eval.to_dict()
    iso.save(str(DATA_DIR / "models" / "isolation_forest.joblib"))

    # --- 2. Autoencoder ---
    logger.info("Training Autoencoder (epochs=%d)...", epochs)
    t0 = time.perf_counter()
    ae = AutoencoderDetector(epochs=epochs, lr=0.001)
    ae.fit(X_train, feature_names, X_val=X_val)
    t_ae = time.perf_counter() - t0
    ae_pred = ae.predict(X_test)
    ae_scores = ae.score_samples(X_test)
    ae_eval = ModelEvaluator.evaluate(y_test, ae_pred, ae_scores)
    results["Autoencoder"] = ae_eval.to_dict()
    ae.save(str(DATA_DIR / "models" / "autoencoder.pt"))

    # --- 3. DBSCAN ---
    logger.info("Training DBSCAN...")
    t0 = time.perf_counter()
    dbscan = DBSCANDetector()
    dbscan.fit(X_train, feature_names, auto_tune=True)
    t_dbscan = time.perf_counter() - t0
    dbscan_pred = dbscan.predict(X_test)
    dbscan_scores = dbscan.score_samples(X_test)
    dbscan_eval = ModelEvaluator.evaluate(y_test, dbscan_pred, dbscan_scores)
    results["DBSCAN"] = dbscan_eval.to_dict()
    dbscan.save(str(DATA_DIR / "models" / "dbscan.joblib"))

    # --- 4. Gaussian Mixture Model (GMM) ---
    logger.info("Training GMM...")
    t0 = time.perf_counter()
    gmm = GMMDetector(n_components=4, random_state=42)
    gmm.fit(X_train, feature_names)
    t_gmm = time.perf_counter() - t0
    gmm_pred = gmm.predict(X_test)
    gmm_scores = gmm.score_samples(X_test)
    gmm_eval = ModelEvaluator.evaluate(y_test, gmm_pred, gmm_scores)
    results["GMM"] = gmm_eval.to_dict()
    gmm.save(str(DATA_DIR / "models" / "gmm.joblib"))

    # --- 5. Four-Model Ensemble ---
    logger.info("Configuring Ensemble and calibrating threshold on Validation set...")
    ensemble = EnsembleDetector()
    ensemble.add_model("isolation_forest", iso)
    ensemble.add_model("autoencoder", ae)
    ensemble.add_model("dbscan", dbscan)
    ensemble.add_model("gmm", gmm)

    ensemble.calibrate_threshold(X_val, y_val)

    t0 = time.perf_counter()
    ens_res = ensemble.predict(X_test)
    t_ensemble_inf = time.perf_counter() - t0

    ens_eval = ModelEvaluator.evaluate(y_test, ens_res.predictions, ens_res.scores)
    results["Ensemble"] = ens_eval.to_dict()

    # --- Print comparison table ---
    eval_objects = {
        name: EvaluationResult(
            precision=d["precision"],
            recall=d["recall"],
            f1=d["f1"],
            auc_roc=d["auc_roc"],
            pr_auc=d.get("pr_auc", 0.0),
            confusion_matrix=np.array(d["confusion_matrix"]),
        )
        for name, d in results.items()
    }
    print("\n" + ModelEvaluator.compare_models(eval_objects))

    total_pipeline_time = time.perf_counter() - t_start_pipeline

    print("\nENSEMBLE SUMMARY")
    print("Weights:")
    for k, v in ensemble.weights.items():
        print(f"  {k} = {v:.4f}")
    print(f"\nValidation threshold:\n  {ensemble.threshold:.4f}")
    print(f"\nTest anomaly count:\n  {int(sum(ens_res.predictions))} / {len(X_test)} samples")
    print(f"\nTIMINGS:")
    print(f"  Data Generation:       {t_gen:.4f}s")
    print(f"  Preprocessing:         {t_prep:.4f}s")
    print(f"  Isolation Forest train:{t_iso:.4f}s")
    print(f"  Autoencoder train:     {t_ae:.4f}s")
    print(f"  DBSCAN train:          {t_dbscan:.4f}s")
    print(f"  GMM train:             {t_gmm:.4f}s")
    print(
        f"  Ensemble inference:    {t_ensemble_inf:.4f}s ({t_ensemble_inf * 1000 / len(X_test):.2f} ms/sample)"
    )
    print(f"  Total pipeline runtime:{total_pipeline_time:.4f}s")

    # --- Save JSON results ---
    out_json = {
        "dataset": {
            "samples": len(df),
            "time_steps": ds_stats["total_samples"],
            "features": len(feature_names),
            "feature_names": feature_names,
            "anomaly_rate": float(df["is_anomaly"].mean()),
            "anomaly_percentage": ds_stats["anomaly_percentage"],
            "anomaly_windows_count": ds_stats["anomaly_windows_count"],
            "min_anomaly_duration": ds_stats["min_anomaly_duration"],
            "max_anomaly_duration": ds_stats["max_anomaly_duration"],
            "avg_anomaly_duration": ds_stats["avg_anomaly_duration"],
            "random_seed": 42,
            "correlations": correlations,
        },
        "split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
            "train_anomalies": ds_stats["train_anomalies"],
            "val_anomalies": ds_stats["val_anomalies"],
            "test_anomalies": ds_stats["test_anomalies"],
        },
        "models": results,
        "ensemble": {
            "weights": ensemble.weights,
            "threshold": ensemble.threshold,
        },
        "timing": {
            "data_generation": t_gen,
            "preprocessing": t_prep,
            "isolation_forest_train": t_iso,
            "autoencoder_train": t_ae,
            "dbscan_train": t_dbscan,
            "gmm_train": t_gmm,
            "ensemble_inference": t_ensemble_inf,
            "total_pipeline": total_pipeline_time,
        },
    }

    EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_JSON, "w") as f:
        json.dump(out_json, f, indent=2)
    logger.info("Saved evaluation JSON to %s", EVAL_JSON)

    # --- Save CSV comparison ---
    csv_rows = []
    for model_name, m_dict in results.items():
        csv_rows.append(
            {
                "Model": model_name,
                "Precision": m_dict["precision"],
                "Recall": m_dict["recall"],
                "F1": m_dict["f1"],
                "ROC-AUC": m_dict["auc_roc"],
                "PR-AUC": m_dict.get("pr_auc", 0.0),
            }
        )
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(EVAL_CSV, index=False)
    logger.info("Saved evaluation CSV to %s", EVAL_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate anomaly detection models")
    parser.add_argument(
        "--data-path", type=Path, default=GENERATED_CSV, help="Path to sensor CSV data"
    )
    parser.add_argument("--days", type=int, default=7, help="Days of data to simulate if generating")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for autoencoder training")
    parser.add_argument(
        "--force-generate", action="store_true", help="Force regeneration of dataset"
    )
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        days=args.days,
        epochs=args.epochs,
        force_generate=args.force_generate,
    )
