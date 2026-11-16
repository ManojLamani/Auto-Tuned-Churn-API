# Auto-Tuned Churn API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg?logo=react)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced Data Science & Machine Learning Decision-Support System**  
> Hyperparameter Optimization with Random Search, Nested Cross-Validation, Production-style FastAPI Inference, and an Analyst React Dashboard.

---

## 📌 Problem Statement

Customer churn presents a significant threat to recurring revenue in telecom and subscription businesses. Retention teams frequently face:
- Massive customer bases with limited capacity for proactive outreach.
- Heuristic or ad-hoc scoring methods with high false-alarm rates.
- Unvalidated models that leak evaluation data or overstate real-world performance due to hyperparameter selection bias.

The **Auto-Tuned Churn API** delivers a reproducible, statistically validated customer-churn prediction and risk-tiering system designed strictly for **decision support**, providing transparency into confidence scores, operating thresholds, and model performance.

---

## 🔬 Scientific Grounding

This project directly implements principles from:

> **Bergstra, J., & Bengio, Y. (2012).**  
> *"Random Search for Hyper-Parameter Optimization."* Journal of Machine Learning Research (JMLR), 13, 281–305.

### Key Methodological Pillars:
1. **Efficiency of Random Search**: Random Search explores high-dimensional parameter spaces with significantly higher sample efficiency than Grid Search because different hyperparameter dimensions have unequal effective impacts on model generalization.
2. **Nested Cross-Validation (Nested CV)**:
   - **Inner CV**: Performs hyperparameter selection and tuning over randomized distributions.
   - **Outer CV**: Estimates the unbiased generalization performance of the tuning procedure itself.
3. **Strict Leakage Prevention**:
   - Reserving an untouched stratified holdout test partition prior to any exploration or transformation.
   - Encapsulating all preprocessing (scaling, imputation, one-hot encoding) strictly inside scikit-learn pipelines fit exclusively on training folds.
4. **Transparent Primary Metric**:
   - Optimizing primarily for **$F_1$-score** to handle target class imbalance (~26.5% churn rate), supported by Precision, Recall, Confusion Matrix, and PR-AUC.
   - Operating threshold selection justified based on business trade-offs (outreach capacity vs. churn capture rate).

---

## 🏛️ System Architecture

```mermaid
flowchart LR
    subgraph Data & Training Pipeline
        A[Telco Churn Dataset] --> B[Data Validation & EDA]
        B --> C[Stratified 80/20 Holdout Split]
        C --> D[Baseline: Logistic Regression]
        C --> E[Candidate: Random Forest + Nested CV]
        E --> F[Single Untouched Test Set Evaluation]
        F --> G[(Versioned Model Artifact)]
    end

    subgraph Production Inference
        G --> H[FastAPI Service /api/v1]
        H <--> I[React + Tailwind UI Dashboard]
    end
```

### High-Level Components:
- **`training/`**: Data ingestion, schema validation, EDA generators, feature engineering pipelines, nested CV orchestration, and experiment logging.
- **`backend/`**: High-performance, stateless FastAPI inference microservice with Pydantic request validation, CORS configuration, and comprehensive health monitoring.
- **`frontend/`**: Analyst-facing React dashboard featuring real-time single-customer scoring, threshold simulators, baseline vs. tuned metric visualizers, and interactive model cards.
- **`artifacts/`**: Versioned, self-contained deployment packages (fitted pipeline + decision threshold + feature schema metadata).

---

## 📂 Project Repository Structure

```text
Auto-Tuned-Churn-API/
├── docs/                     # Architecture, PRD, ADRs, and Model Card
│   ├── PRD.md
│   ├── model-card.md
│   ├── architecture.md
│   └── adr/
├── training/                 # Data science & training pipeline
│   ├── data/                 # Ingestion, validation, splitting
│   ├── eda/                  # Exploratory data analysis scripts
│   ├── features/             # Leakage-safe preprocessing pipelines
│   ├── models/               # Baseline, candidate models, random search, nested CV
│   ├── experiments/          # Structured experiment tracking & logging
│   └── configs/              # Hyperparameter distributions & CV settings
├── backend/                  # FastAPI inference microservice
│   ├── app/
│   │   ├── main.py           # Application entrypoint
│   │   ├── config.py         # App configuration & settings
│   │   ├── schemas.py        # Pydantic input/output schemas
│   │   ├── routes/           # /predict, /model, /experiments, /health
│   │   ├── services/         # Model loader & inference engine
│   │   └── middleware/       # Structured error handling & logging
│   └── tests/                # Unit, schema, and API integration tests
├── frontend/                 # React + Tailwind analyst dashboard
│   ├── src/
│   │   ├── components/       # KPI cards, scoring form, charts, model card
│   │   ├── services/         # API client integration
│   │   └── pages/            # Main dashboard views
│   └── tests/
├── data/                     # Raw & processed data placeholders
├── models/                   # Serialized model artifacts (.joblib)
├── tests/                    # End-to-end integration and data leakage tests
├── scripts/                  # Automation & training execution scripts
├── .github/workflows/        # CI/CD pipelines (lint, test, smoke test)
├── Dockerfile                # Multi-stage container definition
├── docker-compose.yml        # Multi-service container orchestration
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Project documentation (this file)
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- **Python**: 3.14
- **Node.js**: 24 (for the dashboard)
- **Docker**: optional, for the containerized run

### 2. Environment Setup
```bash
git clone https://github.com/ManojLamani/Auto-Tuned-Churn-API.git
cd Auto-Tuned-Churn-API
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Training Pipeline
The EDA and step-by-step analysis live in `notebooks/01_dataset_preparation.ipynb`.
```bash
# Nested CV + Random Search only (prints outer-fold F1 per candidate)
python -m training.models.nested_cv

# Full run: baselines, nested CV, final fit, threshold, one test evaluation,
# then saves models/churn-rf-v1.joblib, model_metadata.json and experiment_log.json
python -m training.models.train_final
```

### 4. Running the Serving Stack
```bash
# FastAPI backend: http://127.0.0.1:8000/docs
uvicorn backend.app.main:app --reload --port 8000

# React dashboard: http://127.0.0.1:5173 (proxies /api to the backend)
cd frontend
npm install
npm run dev
```

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Service and model health (503 when the model is not loaded) |
| POST | `/api/v1/predict` | Churn probability, risk tier, threshold and model version |
| GET | `/api/v1/model` | Model metadata and metrics |
| GET | `/api/v1/experiments` | Tuning experiment log |
| GET | `/api/v1/model-card` | Model card |
| GET | `/api/v1/metrics` | Request/error counts, p50/p95 latency, prediction and risk-tier counts |

`/predict` is rate limited per client (default 60 per minute, set `RATE_LIMIT_PER_MINUTE`); extra requests get 429.

### 5. Docker
```bash
docker compose up -d --build
```
Dashboard on http://localhost:3000, API on http://localhost:8000. Use `API_PORT` / `WEB_PORT` to change the host ports.

### 6. Tests
```bash
pytest -q                     # API, determinism, leakage and nested CV tests
cd frontend && npm test       # UI error-state test
```

---

## 🛡️ Responsible AI & Disclaimer

> [!IMPORTANT]
> **Educational & Decision-Support Prototype**:
> This system is designed solely as a **retention decision-support aid** to help customer success managers prioritize outreach. It is **not** an autonomous cancellation engine, credit assessment tool, or pricing mechanism. Churn predictions represent statistical propensities, not certainties, and should always be accompanied by human judgment.

---

## 📄 License
Distributed under the [MIT License](LICENSE).
