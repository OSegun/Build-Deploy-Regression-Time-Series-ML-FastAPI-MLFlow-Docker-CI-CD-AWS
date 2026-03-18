# Housing Regression MLE — End-to-End Deployment

> **3MTT End-to-End ML Engineering Workshop**
> Predicting US housing prices with a complete machine learning engineering pipeline designed for production deployment and scalability.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [ML Engineering Pipeline](#ml-engineering-pipeline)
- [Model Performance](#model-performance)
- [AWS Cloud Deployment](#aws-cloud-deployment)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

This project walks through building a robust, automated ML system that transforms raw US housing market data into reliable price predictions, accessible through an intuitive web interface. It covers the full ML engineering lifecycle — from raw data ingestion to a scalable cloud-deployed API with a Streamlit dashboard.

---

## Dataset

The project uses a **US housing dataset** with geographic, demographic, and economic features spanning multiple years of market data. It includes 1,000+ data features covering property characteristics, location information, market trends, and temporal factors.

**Download the dataset from Kaggle:**
[https://www.kaggle.com/datasets/shengkunwang/housets-dataset](https://www.kaggle.com/datasets/shengkunwang/housets-dataset)

| Stat | Value |
|---|---|
| Market Price Variance | ~20% across neighborhoods |
| Data Features | 1,000+ geographic, demographic & economic attributes |
| Historical Coverage | 5+ years of market data |

---

## Prerequisites

Before starting, ensure you have the following:

- **IDE** — VSCode, PyCharm, or Cursor
- **Python 3.11** — managed via `uv` (see setup below)
- **Git & GitHub account** — for version control and CI/CD
- **AWS account** — free tier is sufficient for initial deployment
- **Docker Desktop** — for containerisation and local testing
- **Kaggle account** — to download the dataset

---

## Project Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python environment management.

### 1. Install `uv`

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:

```bash
uv --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/<your-org>/Build-Deploy-Regression-Time-Series-ML-FastAPI-MLFlow-Docker-CI-CD-AWS.git
cd Build-Deploy-Regression-Time-Series-ML-FastAPI-MLFlow-Docker-CI-CD-AWS
```

### 3. Scaffold the Project Structure

Run the setup script to create all directories and placeholder files:

```bash
python setup.py
```

This creates the full folder tree (`src/`, `data/`, `docker/`, `infrastructure/`, `tests/`, etc.) with starter files so you can begin building immediately.

### 4. Create a Virtual Environment with Python 3.11

```bash
uv venv --python 3.11 .venv
```

### 5. Activate the Virtual Environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 6. Install Dependencies

Install all packages from `pyproject.toml` using `uv sync` (recommended — resolves and locks the full dependency graph):

```bash
uv sync
```

To also install the dev dependencies (pytest, httpx):

```bash
uv sync --all-extras
```

Alternatively, install directly from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

### 7. Download the Dataset

1. Go to [https://www.kaggle.com/datasets/shengkunwang/housets-dataset](https://www.kaggle.com/datasets/shengkunwang/housets-dataset)
2. Download and extract the CSV file(s) into a `data/raw/` directory:

```bash
mkdir -p data/raw
# Place the downloaded dataset file(s) inside data/raw/
```

### 8. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your AWS credentials, MLflow tracking URI, etc.
```

### 9. Start MLflow Tracking Server (local)

```bash
mlflow ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### 10. Run the Training Pipeline

```bash
python src/pipeline/train.py
```

### 11. Serve the FastAPI Inference API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 12. Launch the Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Project Structure

```
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint, test, build on push/PR
│       └── cd.yml                  # Deploy to AWS ECS on merge to main
├── config/
│   ├── config.yaml                 # Pipeline & model hyperparameter defaults
│   └── great_expectations/         # Data quality suite definitions
├── data/
│   ├── raw/                        # Original downloaded dataset (gitignored)
│   ├── processed/                  # Cleaned & imputed data
│   └── features/                   # Engineered feature sets
├── docker/
│   ├── Dockerfile.api              # FastAPI service image
│   ├── Dockerfile.dashboard        # Streamlit dashboard image
│   └── docker-compose.yml          # Local multi-service orchestration
├── infrastructure/
│   ├── main.tf                     # Terraform root module
│   ├── s3.tf                       # S3 bucket definitions
│   ├── ecs.tf                      # ECS Fargate cluster & task definitions
│   ├── ecr.tf                      # ECR repositories
│   └── alb.tf                      # Application Load Balancer
├── mlruns/                         # MLflow experiment tracking (local)
├── models/
│   └── artifacts/                  # Saved model files (gitignored)
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app entry point
│   │   ├── routers/
│   │   │   └── predict.py          # Prediction endpoints
│   │   └── schemas.py              # Pydantic request/response models
│   ├── dashboard/
│   │   └── app.py                  # Streamlit dashboard
│   ├── pipeline/
│   │   ├── ingest.py               # Data loading & ingestion
│   │   ├── preprocess.py           # Cleaning, imputation, outlier capping
│   │   ├── features.py             # Feature engineering
│   │   ├── train.py                # Model training & Optuna tuning
│   │   └── inference.py            # Batch & real-time prediction
│   └── utils/
│       ├── logging.py              # Centralised logger
│       └── config.py               # Config loader
├── tests/
│   ├── test_api.py                 # FastAPI endpoint tests
│   ├── test_pipeline.py            # Pipeline unit tests
│   └── test_features.py            # Feature engineering tests
├── .env.example                    # Environment variable template
├── .gitignore
├── .python-version                 # Pins Python 3.11 for uv
├── pyproject.toml                  # Project metadata & tool config
├── requirements.txt                # Pinned dependencies
├── uv.lock                         # uv lockfile
└── README.md
```

---

## Technology Stack

### Machine Learning

| Library | Purpose |
|---|---|
| XGBoost | Gradient boosting regression on tabular data |
| Optuna | Bayesian hyperparameter optimisation |
| MLflow | Experiment tracking & model versioning |
| Scikit-learn | Preprocessing, feature engineering & evaluation |
| Great Expectations | Automated data quality checks |
| Pytest | Unit testing & code quality |

### API & Dashboard

| Library | Purpose |
|---|---|
| FastAPI | High-performance REST API with auto docs |
| Pydantic | Data validation & settings management |
| Streamlit | Interactive web dashboard |
| Plotly | Responsive, interactive visualisations |

### Data Processing

| Library | Purpose |
|---|---|
| Pandas | Data manipulation & transformation |
| NumPy | Numerical computing |
| Dask | Parallel computing on large datasets |

### Cloud & DevOps

| Tool | Purpose |
|---|---|
| AWS S3 | Centralised data lake & model storage |
| AWS ECR | Container image registry |
| AWS ECS Fargate | Serverless container orchestration |
| AWS ALB | Application load balancing |
| Docker | Containerisation & reproducibility |
| GitHub Actions | CI/CD automation |

---

## ML Engineering Pipeline

```
Raw Data
   │
   ▼
1. Data Loading & Ingestion       ← Automated extraction, schema validation
   │
   ▼
2. Data Preprocessing             ← Great Expectations, imputation, outlier capping
   │
   ▼
3. Feature Engineering            ← Location clusters, price/sqft, temporal patterns
   │
   ▼
4. Model Training & Tuning        ← XGBoost + Optuna + MLflow experiment tracking
   │
   ▼
5. Pipeline Orchestration         ← Modular, reusable, reproducible components
   │
   ▼
6. Containerisation & CI/CD       ← Docker + GitHub Actions → AWS ECR
   │
   ▼
7. Model Deployment               ← FastAPI on AWS ECS Fargate + ALB
   │
   ▼
8. Frontend Dashboard             ← Streamlit with Plotly visualisations
```

### Key Data Engineering Practices

- **Time-aware splits** — Chronological train/validation/test splits to prevent data leakage
- **Robust preprocessing** — Median/mode imputation, IQR outlier capping, target-aware encoding for high-cardinality categoricals
- **Feature engineering** — Geographic k-means clusters, property age, price-per-square-foot proxy, seasonal indicators

---

## Model Performance

Validated across multiple time periods using time-based cross-validation:

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | $12,450 |
| RMSE (Root Mean Squared Error) | $18,230 |
| Mean Percentage Error | 8.2% |

---

## AWS Cloud Deployment

| Service | Role |
|---|---|
| S3 | Raw datasets, processed features, trained models, batch outputs |
| ECR | Container image storage with versioning |
| ECS Fargate | Serverless container orchestration with auto-scaling |
| ALB | Routes traffic to containers with SSL termination |

Infrastructure is defined as code (Terraform / AWS CloudFormation) for reproducible, version-controlled deployments with support for blue-green zero-downtime updates.

---

## Future Enhancements

1. **Production Domain** — Custom domain with SSL for HTTPS deployment
2. **Monitoring & Observability** — Real-time drift detection, logging, distributed tracing
3. **A/B Testing Framework** — Statistical significance testing for model version comparison
4. **Automated Retraining** — CI/CD-triggered retraining on performance degradation or data drift
5. **Scaling & Cost Optimisation** — Multi-region deployment, auto-scaling policies for traffic-based compute adjustment

---

## Workshop Reference

This project accompanies the **Housing Regression MLE End-to-End Deployment** slide deck provided to workshop participants. Refer to the PDF for detailed explanations of each pipeline stage, architecture decisions, and deployment strategies.