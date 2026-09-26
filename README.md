<div align="center">

# 🏭 Industrial IoT Anomaly Intelligence Platform

**A production-grade, unsupervised anomaly detection system for Industrial IoT telemetry**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-199%20passed-brightgreen?logo=pytest)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Detect anomalies in complex industrial process telemetry using a four-model ensemble — with full explainability, real-time streaming replay, interactive operator dashboard, and REST API.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo — What the Dashboard Does](#-live-demo--what-the-dashboard-does)
- [Architecture](#-architecture)
- [Dataset & Sensor Inventory](#-dataset--sensor-inventory)
- [ML Models & Ensemble](#-ml-models--ensemble)
- [Evaluation Results](#-evaluation-results)
- [Explainability Engine](#-explainability-engine)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Running the Dashboard](#-running-the-dashboard)
- [REST API Reference](#-rest-api-reference)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Docker](#-docker)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)

---

## 🔍 Overview

This project implements a **complete, end-to-end unsupervised anomaly detection pipeline** for an industrial water treatment and pumping facility simulation. It ingests multivariate IoT telemetry from 28 correlated sensors, runs four independently trained anomaly detectors, fuses their outputs through a calibrated ensemble, explains every alert with sensor-level root cause analysis, and surfaces the results through a professional interactive operator dashboard and a REST API.

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No Labels Required** | Fully unsupervised — trains only on normal operational data |
| **Leak-Safe Evaluation** | Strict chronological 70/15/15 split; threshold calibrated on validation only |
| **Explainable by Default** | Every alert comes with ranked sensor contributors, z-scores, and correlation context |
| **Physics-Correlated Simulation** | 28 sensors coupled via real process physics (pump speed → flow → pressure → power) |
| **Production-Ready Architecture** | REST API, SQLite persistence, async streaming, Docker support, CI/CD |

---

## 🖥️ Live Demo — What the Dashboard Does

The interactive Streamlit dashboard (`src/dashboard/app.py`) is an **operator-facing industrial monitoring application**, not a static report. The operator workflow is:

```
CONFIGURE → START MONITORING → INGEST TELEMETRY → DETECT ANOMALY
    ↓
ALERT OPERATOR → INVESTIGATE → EXPLAIN → TAKE ACTION
    ↓
REPLAY / REVIEW → GENERATE REPORT
```

### Dashboard Pages

| Section | What You Can Do |
|---------|----------------|
| **⚡ Overview** | Live KPI strip, ensemble score gauge, 7-subsystem plant pulse, anomaly timeline, root-cause panel |
| **🚨 Anomaly Center** | Full incident ledger, filterable event log, CSV export, alert resolution |
| **🔬 Sensor Explorer** | Per-sensor time-series with ±2σ normal envelope, Z-score overlay, dynamic time windows |
| **🧠 Model Intelligence** | Individual detector cards (algorithm + current score), model agreement matrix, radar comparison |
| **⚡ Stream Replay** | Configurable live stream simulation (1×–50× speed), latency benchmark, non-blocking alerts |
| **🕸️ Correlation Map** | Interactive 28×28 Pearson correlation heatmap, per-sensor coupling list |
| **🖥️ System Status** | Architecture specs, dataset partition breakdown, test verification ledger |
| **📖 About / Methodology** | Full pipeline explanation, academic context, algorithm descriptions |

### Demo Controls

- **📽️ Presentation Mode** — enlarges primary charts, hides technical panels for projector/classroom use
- **⏩ Jump to Next Anomaly** — steps through verified ground-truth anomaly events instantly for live demos

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  Physics-Correlated IoT Generator (28 sensors, 7 groups)    │
│  10 anomaly types · 7-day benchmark · 10,080 samples        │
└──────────────────────┬──────────────────────────────────────┘
                       │ chronological split (70/15/15)
┌──────────────────────▼──────────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│  Standard Scaler · rolling statistics · sliding window       │
└──────────┬───────────────────────────────────────────────────┘
           │  normal training data only
    ┌──────▼──────────────────────────────────┐
    │           FOUR-MODEL ENSEMBLE           │
    │  ┌────────────┐   ┌──────────────────┐  │
    │  │ Isolation  │   │   PyTorch Deep   │  │
    │  │   Forest   │   │   Autoencoder    │  │
    │  └────────────┘   └──────────────────┘  │
    │  ┌────────────┐   ┌──────────────────┐  │
    │  │   DBSCAN   │   │ Gaussian Mixture │  │
    │  │ (density)  │   │    Model (GMM)   │  │
    │  └────────────┘   └──────────────────┘  │
    │         ↓  score normalization ↓         │
    │    Equal-Weight Ensemble (0.25 each)     │
    │    Validation-Calibrated Threshold       │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │         EXPLAINABILITY ENGINE           │
    │  z-score deviation · Pearson r           │
    │  model consensus · trajectory trend     │
    └──────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │            PRESENTATION LAYER           │
    │  Streamlit Dashboard  │  FastAPI REST   │
    │  Streaming Replay     │  SQLite Store   │
    └─────────────────────────────────────────┘
```

---

## 📡 Dataset & Sensor Inventory

### Synthetic Industrial Benchmark

A physics-correlated process simulation of a **multi-stage water treatment and pumping facility** at 1-minute sampling intervals.

| Property | Value |
|----------|-------|
| Total samples | 10,080 (7 days × 24 h × 60 min) |
| Sensors | 28 correlated process variables |
| Subsystems | 7 industrial groups |
| Anomaly rate | ~4% injected |
| Train split | 70% (7,056 samples) — **normal only** |
| Validation split | 15% (1,512 samples) |
| Test split | 15% (1,512 samples) |
| Random seed | 42 (fully deterministic) |

### Sensor Groups & Variables

| Subsystem | Sensors | Units |
|-----------|---------|-------|
| **Actuator / Control** | `pump_1_speed`, `pump_2_speed`, `valve_1_position`, `valve_2_position`, `dosing_rate` | RPM, %, L/h |
| **Flow** | `pump_flow`, `inlet_flow`, `outlet_flow`, `recycle_flow` | m³/h |
| **Pressure** | `pressure`, `inlet_pressure`, `pump_pressure`, `filter_pressure` | bar |
| **Level** | `tank_1_level`, `tank_2_level`, `tank_3_level`, `reservoir_level` | m |
| **Temperature** | `temperature`, `pump_temperature`, `motor_temperature` | °C |
| **Process Quality** | `pH`, `conductivity`, `turbidity`, `chlorine` | pH, µS/cm, NTU, mg/L |
| **Electrical / Motor** | `power`, `motor_current`, `motor_voltage`, `vibration` | kW, A, V, mm/s |

### Operating Regimes

Three smooth, continuously transitioning operating modes are simulated:

- **Low Load** — reduced flow, lower pressure, lower power consumption
- **Normal Load** — baseline operating point
- **High Load** — maximum throughput, elevated motor temperatures

### Injected Anomaly Types

| # | Type | Description |
|---|------|-------------|
| 1 | `sudden_spike` | Instantaneous large positive deviation |
| 2 | `sudden_drop` | Instantaneous large negative deviation |
| 3 | `gradual_drift` | Slow monotonic shift over 30–90 min |
| 4 | `sensor_bias` | Persistent fixed offset |
| 5 | `increased_noise` | Elevated stochastic variance |
| 6 | `sensor_dropout` | Complete signal loss / stuck at zero |
| 7 | `stuck_sensor` | Signal frozen at last value |
| 8 | `oscillation` | High-frequency sinusoidal pattern |
| 9 | `correlated_multi_sensor` | Simultaneous deviation across coupled sensors |
| 10 | `gradual_process_degradation` | Multi-sensor slow decay mimicking equipment wear |

---

## 🧠 ML Models & Ensemble

### Model Descriptions

#### 1. Isolation Forest
- **Algorithm**: Random recursive partitioning of feature space; anomalies are isolated in fewer splits
- **Config**: `n_estimators=100`, `contamination=0.05`
- **Strengths**: Fast, scales well to high dimensions, effective for point anomalies
- **Output**: Anomaly score from decision function (normalized to [0, 1])

#### 2. PyTorch Deep Autoencoder
- **Architecture**: Encoder `28 → 32 → 16 → 8` / Decoder `8 → 16 → 32 → 28`
- **Training**: 30 epochs, Adam optimizer, MSE reconstruction loss, normal samples only
- **Strengths**: Learns latent normal manifold; high reconstruction error signals anomaly
- **Output**: Per-sample mean squared reconstruction error (normalized to [0, 1])

#### 3. DBSCAN (Density-Based Spatial Clustering)
- **Algorithm**: Groups dense neighborhoods; points outside all clusters = anomalies
- **Config**: `eps=1.9`, `min_samples=5`
- **Strengths**: No distributional assumptions; catches collective/contextual anomalies
- **Output**: Binary label (−1 = anomaly), converted to 0/1 score

#### 4. Gaussian Mixture Model (GMM)
- **Algorithm**: Fits a mixture of Gaussians to normal data; low log-likelihood = anomaly
- **Config**: `n_components` tuned on validation data
- **Strengths**: Captures multimodal normal distributions (low/normal/high load regimes)
- **Output**: Negative log-likelihood (normalized to [0, 1])

### Ensemble Strategy

```
ensemble_score = 0.25 × IF_score
               + 0.25 × AE_score
               + 0.25 × DBSCAN_score
               + 0.25 × GMM_score

flag_as_anomaly = ensemble_score >= threshold*
```

> \* Threshold determined by **F1-maximization grid search on the validation set** — no test data leakage.

---

## 📊 Evaluation Results

> Evaluated on the held-out **test split (15%, 1,512 samples)** — never used during training or threshold calibration.

| Metric | Value |
|--------|-------|
| **Precision** | 0.8211 |
| **Recall** | 0.4756 |
| **F1 Score** | 0.6023 |
| **Ensemble Inference** | ~0.71 ms/sample |
| **Streaming Latency** | ~54 ms/sample (target: <100 ms) |

> The calibrated threshold and full metric breakdown are saved to `data/evaluation_results.json` after every training run.

---

## 🔬 Explainability Engine

Every detected anomaly is accompanied by a structured **AnomalyExplanation** object (`src/analysis/root_cause.py`) that contains:

- **Top contributing sensors** ranked by absolute z-score deviation from the normal training baseline
- **Per-sensor statistics**: mean deviation, normalized deviation (σ), percentage deviation, direction (`Above Normal` / `Below Normal`), and recent trajectory trend
- **Empirical Pearson correlations** between all sensor pairs — highlights co-moving process variables without claiming causality
- **Model consensus**: which of the 4 detectors flagged this sample and with what score
- **Human-readable narrative** describing what was statistically unusual, in plain language

> **Important**: Diagnostic language is strictly statistical. Equipment failure attribution and cyberattack claims require domain engineer verification.

---

## 📁 Project Structure

```
iot-anomaly-detection/
│
├── src/                         # All application source code
│   ├── __init__.py
│   ├── main.py                  # Application entry point
│   ├── api/                     # FastAPI REST layer
│   │   └── routes.py            # All HTTP endpoints
│   ├── alerting/                # Alert rules engine
│   │   ├── engine.py            # Score/frequency/cooldown logic
│   │   └── notifier.py          # Notification channels
│   ├── analysis/                # Root cause analysis
│   │   └── root_cause.py        # AnomalyExplanation, RootCauseAnalyzer
│   ├── dashboard/               # Streamlit interactive dashboard
│   │   ├── app.py               # Main dashboard entry point
│   │   ├── components.py        # Reusable chart/UI components
│   │   ├── state.py             # Session state management
│   │   └── theme.py             # Design tokens, CSS injection
│   ├── data/                    # Data pipeline
│   │   ├── generator.py         # Physics-correlated IoT data generator
│   │   ├── loader.py            # Dataset loading and validation
│   │   └── preprocessor.py      # Feature scaling, splitting
│   ├── models/                  # Anomaly detectors
│   │   ├── base.py              # Abstract detector interface
│   │   ├── isolation_forest.py  # Isolation Forest wrapper
│   │   ├── autoencoder.py       # PyTorch autoencoder
│   │   ├── dbscan.py            # DBSCAN wrapper
│   │   ├── gmm.py               # Gaussian Mixture Model
│   │   └── ensemble.py          # Score fusion + threshold calibration
│   ├── streaming/               # Real-time streaming
│   │   ├── pipeline.py          # Async streaming pipeline
│   │   └── replay.py            # Chronological dataset replay
│   └── utils/                   # Shared utilities
│       ├── config.py            # YAML configuration loader
│       ├── database.py          # SQLite operations
│       └── logging.py           # Structured logging
│
├── tests/                       # Full test suite (199 tests)
│   ├── test_data_generator.py
│   ├── test_models.py
│   ├── test_ensemble.py
│   ├── test_explainability.py
│   ├── test_streaming.py
│   ├── test_alerting.py
│   ├── test_api.py
│   └── test_dashboard.py
│
├── scripts/                     # CLI utilities
│   ├── train_and_evaluate.py    # Full training + evaluation pipeline
│   ├── populate_db.py           # Populate SQLite from generated data
│   ├── replay_demo.py           # Streaming latency benchmark
│   └── verify_interactive_app.py # End-to-end app verification
│
├── configs/
│   └── config.yaml              # All tunable parameters
│
├── data/                        # Generated data (git-ignored)
│   ├── generated/               # sensor_data.csv, labels.csv
│   ├── models/                  # Saved model files (.pkl, .pt)
│   └── evaluation_results.json  # Latest benchmark metrics
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
│
├── .streamlit/
│   └── config.toml              # Streamlit server settings
│
├── Dockerfile
├── docker-compose.yml
├── Makefile                     # Developer shortcuts
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- `pip` or a virtual environment manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/n-a-n-d-a-n/iot-anomaly-ml.git
cd iot-anomaly-ml
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the Synthetic Dataset

```bash
python scripts/train_and_evaluate.py --generate-only
```

This generates 10,080 samples (7-day benchmark) with 28 correlated sensors and injects 10 anomaly types. Output is saved to `data/generated/`.

### 5. Train All Four Models

```bash
python scripts/train_and_evaluate.py
```

This trains Isolation Forest, Autoencoder, DBSCAN, and GMM; calibrates the ensemble threshold on the validation set; and saves evaluation results to `data/evaluation_results.json`.

### 6. Populate the Database

```bash
python scripts/populate_db.py
```

Runs the full dataset through the trained ensemble and stores results in `data/anomaly_detection.db`.

---

## 🖥️ Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First-time setup**: Make sure you have completed all Quick Start steps (generate → train → populate-db) before launching the dashboard.

### Alternative: Makefile Shortcuts

```bash
make install        # Install dependencies
make generate-data  # Generate synthetic IoT dataset
make train          # Train all models + calibrate ensemble
make populate-db    # Populate SQLite database
make dashboard      # Launch Streamlit dashboard
make run            # Launch FastAPI server
make test           # Run full test suite
make lint           # Lint with ruff
make lint-fix       # Auto-fix lint issues
```

---

## 🌐 REST API Reference

Start the API server:

```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check / readiness probe |
| `POST` | `/score` | Score a single sensor reading |
| `POST` | `/score/batch` | Score a batch of readings |
| `GET` | `/alerts` | Retrieve alert history (filterable by severity, time) |
| `PUT` | `/alerts/{id}/resolve` | Mark an alert as resolved |
| `GET` | `/models/status` | Model load status and metadata |
| `GET` | `/sensors/health` | Per-sensor health summary |
| `GET` | `/metrics` | Current benchmark evaluation results |
| `GET` | `/sensors` | All 28 sensors and subsystem groupings |
| `GET` | `/anomalies` | Query anomaly event logs from the database |
| `GET` | `/explanation/{timestamp}` | Detailed explainability diagnostics for an event |

### Example: Score a Sample

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-01T12:00:00Z",
    "sensor_readings": {
      "pump_1_speed": 1450.0,
      "pump_flow": 62.5,
      "pressure": 3.8,
      "temperature": 28.0
    }
  }'
```

### Example Response

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "ensemble_score": 0.743,
  "is_anomaly": true,
  "threshold": 0.606,
  "model_scores": {
    "isolation_forest": 0.68,
    "autoencoder": 0.81,
    "dbscan": 0.75,
    "gmm": 0.72
  }
}
```

---

## ⚙️ Configuration

All parameters are controlled from [`configs/config.yaml`](configs/config.yaml):

```yaml
generator:
  seed: 42                    # Deterministic generation
  days: 7                     # Benchmark length
  sampling_interval_minutes: 1
  anomaly_rate: 0.04

models:
  isolation_forest:
    contamination: 0.05
    n_estimators: 100
  autoencoder:
    hidden_dims: [32, 16, 8]
    epochs: 30
    lr: 0.001
  dbscan:
    eps: 1.9
    min_samples: 5
  ensemble_weights:
    isolation_forest: 0.25
    autoencoder: 0.25
    dbscan: 0.25
    gmm: 0.25

streaming:
  queue_size: 1000
  window_minutes: 60
  latency_target_ms: 50

alerting:
  score_threshold: 0.8
  cooldown_seconds: 300
```

---

## 🧪 Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test module
python -m pytest tests/test_models.py -v

# Run the end-to-end verification script
python scripts/verify_interactive_app.py
```

### Test Coverage Summary

| Module | Tests |
|--------|-------|
| Data generator | ✅ |
| ML models (IF, AE, DBSCAN, GMM) | ✅ |
| Ensemble + threshold calibration | ✅ |
| Explainability / root cause | ✅ |
| Streaming pipeline | ✅ |
| Alerting engine | ✅ |
| REST API | ✅ |
| Dashboard components | ✅ |
| **Total** | **199 tests, 0 failures** |

---

## 🐳 Docker

```bash
# Build and start both API + dashboard services
docker-compose up --build

# Stop services
docker-compose down
```

| Service | URL |
|---------|-----|
| FastAPI REST API | http://localhost:8000 |
| Streamlit Dashboard | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **ML / AI** | scikit-learn, PyTorch, NumPy, SciPy |
| **Data** | Pandas, NumPy |
| **Dashboard** | Streamlit, Plotly |
| **REST API** | FastAPI, Uvicorn |
| **Database** | SQLite, aiosqlite |
| **Configuration** | PyYAML |
| **Testing** | pytest, pytest-cov, pytest-asyncio |
| **Code Quality** | Ruff, pre-commit |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker, Docker Compose |

---

## 🗺️ Roadmap

- [ ] **SWaT / WADI integration** — plug in real physical testbed data when access is approved
- [ ] **Online adaptive retraining** — concept drift detection + incremental model updates
- [ ] **MQTT / Kafka ingest** — replace file-based replay with live broker subscription
- [ ] **Multi-facility federation** — extend to N parallel plant instances
- [ ] **Mobile-responsive dashboard** — operator alerts on handheld devices
- [ ] **Anomaly similarity search** — retrieve historical incidents matching the current pattern

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for Industrial IoT Anomaly Detection**

*If this project helped you, consider giving it a ⭐*

</div>

