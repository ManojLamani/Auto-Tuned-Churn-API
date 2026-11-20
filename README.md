# Auto-Tuned Churn API

A **customer churn prediction service** with an auto-tuned Random Forest, a FastAPI inference API and a React dashboard for retention analysts. Enter a customer's account details, and the app returns their churn probability, a risk tier and the model version behind the score.

**Live demo:** [auto-tuned-churn-api.vercel.app](https://auto-tuned-churn-api.vercel.app)

---

## ✨ Features

- **Auto-Tuned Model** – Random Search ([Bergstra & Bengio, 2012](https://www.jmlr.org/papers/v13/bergstra12a.html)) tunes a Random Forest, and nested cross-validation measures how well the tuning generalises.
- **Leakage-Safe Training** – All preprocessing runs inside scikit-learn pipelines, and the holdout test set is used only once.
- **Customer Scoring** – Score any customer from the dashboard and get a churn probability with a High, Medium or Low risk tier.
- **Model Transparency** – Model performance, the tuning experiment log and a model card are all visible in the dashboard.
- **Versioned Releases** – Every training run saves a new model version with a checksum manifest, and you can switch back to any earlier version.
- **Production-Style API** – Input validation, rate limiting, structured errors, health checks and request metrics.

---

## 📊 Results

Measured on the IBM Telco holdout test set:

| Model | F1 | Recall | ROC-AUC |
|-------|----|--------|---------|
| Logistic Regression (baseline) | 0.604 | 0.559 | 0.842 |
| **Random Forest (tuned)** | **0.636** | **0.767** | 0.839 |

Nested cross-validation F1: **0.631 ± 0.013** (5 outer folds, 3 inner folds).

---

## 🗂 File Structure

```
Auto-Tuned-Churn-API/
├── api/
│   └── index.py          # Vercel entrypoint (loads the FastAPI app and model)
├── backend/app/
│   ├── main.py           # FastAPI app and /api/v1 endpoints
│   ├── predictor.py      # Model loading and inference
│   ├── schemas.py        # Request and response schemas
│   └── monitoring.py     # Request metrics and rate limiting
├── training/
│   ├── models/           # Nested CV and final training scripts
│   ├── features.py       # Preprocessing pipelines
│   ├── artifacts.py      # Model releases and version switching
│   └── validate_artifact.py
├── frontend/src/
│   ├── pages/            # Overview, Score, Experiments, Model card
│   └── api.js            # API client
├── models/               # Saved model versions and experiment logs
├── notebooks/            # EDA and training walkthrough
├── tests/                # API, data, leakage and performance tests
├── vercel.json           # Vercel deployment config
├── docker-compose.yml    # Local API + dashboard containers
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites
- **Python 3.14**. Check with `python --version` (Windows: `py --version`).
- **Node.js 24** for the dashboard. Check with `node --version`.
- **Docker** (optional), to run everything with one command.

### 1. Clone the Repository
```bash
git clone https://github.com/ManojLamani/Auto-Tuned-Churn-API.git
cd Auto-Tuned-Churn-API
```

### 2. Create a Virtual Environment and Install Dependencies

#### 🐧 Linux & 🍎 macOS
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

#### 🪟 Windows (PowerShell)
```powershell
py -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

### 3. Run the API

**Linux & macOS** (with the venv activated):
```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

**Windows:**
```powershell
venv\Scripts\python -m uvicorn backend.app.main:app --reload --port 8000
```

The API docs will be available at **http://127.0.0.1:8000/docs**.

### 4. Run the Dashboard
In a second terminal:
```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at **http://127.0.0.1:5173**.

### Using Docker Instead
```bash
docker compose up -d --build
```
The dashboard runs on **http://localhost:3000** and the API on **http://localhost:8000**.

### Troubleshooting
- **`'uvicorn' is not recognized`:** run it through Python as shown above (`python -m uvicorn …`).
- **Port 8000 or 3000 already in use:** for uvicorn, use `--port 8001`. For Docker, set `API_PORT` and `WEB_PORT`, e.g. `API_PORT=8001 WEB_PORT=3001 docker compose up -d`.
- **Dashboard shows "Service unavailable":** the API is not running. Start it (step 3), and the dashboard reconnects automatically.

---

## 🚀 How to Use

### Overview
Open the dashboard. The **Overview** page shows the tuned model's test scores next to the baseline, the nested CV score and the model comparison.

### Score a Customer
1. Click **Score customer** in the sidebar.
2. Fill in the customer's details, or click **High-risk example** or **Low-risk example** to load a sample.
3. Click **Score customer**. The result shows the churn probability, the risk tier and the model version.

### Experiments & Model Card
- **Experiments** lists the tuning runs with their CV F1 scores, plus the nested CV score for each outer fold.
- **Model card** describes the model, its data, evaluation, threshold and limitations.

### Training a New Model
```bash
python -m training.models.train_final      # train, evaluate and save a new version
python -m training.artifacts churn-rf-v1   # switch the active model version
python -m training.validate_artifact       # verify the active version
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Service and model status |
| POST | `/api/v1/predict` | Churn probability, risk tier and model version |
| GET | `/api/v1/model` | Model metadata and metrics |
| GET | `/api/v1/experiments` | Tuning experiment log |
| GET | `/api/v1/model-card` | Model card |
| GET | `/api/v1/metrics` | Request, latency and prediction counters |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Machine Learning** | scikit-learn, pandas, NumPy |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | React 19, Vite, Tailwind CSS |
| **Testing** | pytest, Vitest, Playwright |
| **Deployment** | Vercel, Docker, GitHub Actions |

---

## 📝 Notes

- The live demo runs on **Vercel**: the dashboard is a static site and the API is a Python serverless function. Every push to `main` redeploys it.
- `/predict` is limited to 60 requests per minute per client (`RATE_LIMIT_PER_MINUTE`).
- Training can optionally add synthetic rows from [Kaggle Playground S6E3](https://www.kaggle.com/competitions/playground-series-s6e3/data) placed at `data/kaggle_s6e3_train.csv`. They are used for training only, never for validation or testing.
- Run the tests with `pytest -q`, and `npm test` inside `frontend/`.
- This is a **decision-support tool**. Predictions are probabilities, not certainties, and should be reviewed by a person before any action is taken.
