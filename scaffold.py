"""
Project scaffolding script — Housing Regression MLE End-to-End Deployment.

Run this once after cloning to create the full folder structure and placeholder files.

Usage:
    python scaffold.py
"""

from pathlib import Path

BASE = Path(__file__).parent

DIRS = [
    ".github/workflows",
    "config/great_expectations",
    "data/raw",
    "data/processed",
    "data/features",
    "docker",
    "infrastructure",
    "mlruns",
    "models/artifacts",
    "notebooks",
    "src/api/routers",
    "src/dashboard",
    "src/pipeline",
    "src/utils",
    "tests",
]

FILES = {
    # ── Environment & config ──────────────────────────────────────────────────
    ".python-version": "3.11\n",
    ".env.example": (
        "# AWS Credentials\n"
        "AWS_ACCESS_KEY_ID=your-access-key\n"
        "AWS_SECRET_ACCESS_KEY=your-secret-key\n"
        "AWS_DEFAULT_REGION=us-east-1\n"
        "\n"
        "# S3\n"
        "S3_BUCKET_NAME=your-housing-ml-bucket\n"
        "\n"
        "# MLflow\n"
        "MLFLOW_TRACKING_URI=http://localhost:5000\n"
        "MLFLOW_EXPERIMENT_NAME=housing-regression\n"
        "\n"
        "# API\n"
        "API_HOST=0.0.0.0\n"
        "API_PORT=8000\n"
    ),
    "config/config.yaml": (
        "data:\n"
        "  raw_path: data/raw\n"
        "  processed_path: data/processed\n"
        "  features_path: data/features\n"
        "\n"
        "model:\n"
        "  name: xgboost_housing\n"
        "  target_column: price\n"
        "  test_size: 0.2\n"
        "  val_size: 0.1\n"
        "\n"
        "optuna:\n"
        "  n_trials: 50\n"
        "  timeout: 600\n"
        "\n"
        "mlflow:\n"
        "  experiment_name: housing-regression\n"
    ),
    # ── GitHub Actions ────────────────────────────────────────────────────────
    ".github/workflows/ci.yml": (
        "name: CI\n"
        "\n"
        "on:\n"
        "  push:\n"
        "    branches: [main, develop]\n"
        "  pull_request:\n"
        "    branches: [main]\n"
        "\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: astral-sh/setup-uv@v3\n"
        "        with:\n"
        "          python-version: '3.11'\n"
        "      - run: uv sync --all-extras\n"
        "      - run: uv run pytest tests/\n"
    ),
    ".github/workflows/cd.yml": (
        "name: CD\n"
        "\n"
        "on:\n"
        "  push:\n"
        "    branches: [main]\n"
        "\n"
        "jobs:\n"
        "  deploy:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - name: Configure AWS credentials\n"
        "        uses: aws-actions/configure-aws-credentials@v4\n"
        "        with:\n"
        "          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}\n"
        "          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}\n"
        "          aws-region: ${{ secrets.AWS_DEFAULT_REGION }}\n"
        "      - name: Build and push Docker image\n"
        "        run: |\n"
        "          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY\n"
        "          docker build -f docker/Dockerfile.api -t $ECR_REGISTRY/$ECR_REPO:latest .\n"
        "          docker push $ECR_REGISTRY/$ECR_REPO:latest\n"
    ),
    # ── Docker ────────────────────────────────────────────────────────────────
    "docker/Dockerfile.api": (
        "FROM python:3.11-slim\n"
        "\n"
        "WORKDIR /app\n"
        "\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir uv && uv pip install --system -r requirements.txt\n"
        "\n"
        "COPY src/ ./src/\n"
        "COPY config/ ./config/\n"
        "COPY models/ ./models/\n"
        "\n"
        "EXPOSE 8000\n"
        "\n"
        'CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    ),
    "docker/Dockerfile.dashboard": (
        "FROM python:3.11-slim\n"
        "\n"
        "WORKDIR /app\n"
        "\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir uv && uv pip install --system -r requirements.txt\n"
        "\n"
        "COPY src/dashboard/ ./src/dashboard/\n"
        "COPY config/ ./config/\n"
        "\n"
        "EXPOSE 8501\n"
        "\n"
        'CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]\n'
    ),
    "docker/docker-compose.yml": (
        "version: '3.9'\n"
        "\n"
        "services:\n"
        "  api:\n"
        "    build:\n"
        "      context: ..\n"
        "      dockerfile: docker/Dockerfile.api\n"
        "    ports:\n"
        "      - '8000:8000'\n"
        "    env_file: ../.env\n"
        "\n"
        "  dashboard:\n"
        "    build:\n"
        "      context: ..\n"
        "      dockerfile: docker/Dockerfile.dashboard\n"
        "    ports:\n"
        "      - '8501:8501'\n"
        "    env_file: ../.env\n"
        "    depends_on:\n"
        "      - api\n"
        "\n"
        "  mlflow:\n"
        "    image: ghcr.io/mlflow/mlflow:v2.13.0\n"
        "    ports:\n"
        "      - '5000:5000'\n"
        "    command: mlflow server --host 0.0.0.0 --port 5000\n"
    ),
    # ── Terraform stubs ───────────────────────────────────────────────────────
    "infrastructure/main.tf": "# TD: Terraform root module — provider config and backend\n",
    "infrastructure/s3.tf": "# TD: Define S3 bucket resources for data lake and model storage\n",
    "infrastructure/ecs.tf": "# TD: Define ECS Fargate cluster and task definitions\n",
    "infrastructure/ecr.tf": "# TD: Define ECR repositories for API and dashboard images\n",
    "infrastructure/alb.tf": "# TD: Define Application Load Balancer and target groups\n",
    # ── Source packages ───────────────────────────────────────────────────────
    "src/__init__.py": "",
    "src/api/__init__.py": "",
    "src/api/routers/__init__.py": "",
    "src/dashboard/__init__.py": "",
    "src/pipeline/__init__.py": "",
    "src/utils/__init__.py": "",
    "tests/__init__.py": "",
    "src/api/main.py": (
        "from fastapi import FastAPI\n"
        "from src.api.routers import predict\n"
        "\n"
        'app = FastAPI(title="Housing Price Prediction API", version="0.1.0")\n'
        "\n"
        'app.include_router(predict.router, prefix="/predict", tags=["predictions"])\n'
        "\n"
        "\n"
        '@app.get("/health")\n'
        "def health_check():\n"
        '    return {"status": "ok"}\n'
    ),
    "src/api/schemas.py": (
        "from pydantic import BaseModel\n"
        "\n"
        "\n"
        "class PredictionRequest(BaseModel):\n"
        "    bedrooms: int\n"
        "    bathrooms: float\n"
        "    sqft_living: float\n"
        "    sqft_lot: float\n"
        "    floors: float\n"
        "    waterfront: int\n"
        "    view: int\n"
        "    condition: int\n"
        "    grade: int\n"
        "    yr_built: int\n"
        "    zipcode: str\n"
        "    lat: float\n"
        "    long: float\n"
        "\n"
        "\n"
        "class PredictionResponse(BaseModel):\n"
        "    predicted_price: float\n"
        "    confidence_interval: dict\n"
    ),
    "src/api/routers/predict.py": (
        "from fastapi import APIRouter\n"
        "from src.api.schemas import PredictionRequest, PredictionResponse\n"
        "\n"
        "router = APIRouter()\n"
        "\n"
        "\n"
        '@router.post("/", response_model=PredictionResponse)\n'
        "def predict(request: PredictionRequest):\n"
        "    # TD: load model and run inference\n"
        "    return PredictionResponse(\n"
        "        predicted_price=0.0,\n"
        '        confidence_interval={"lower": 0.0, "upper": 0.0},\n'
        "    )\n"
    ),
    "src/dashboard/app.py": (
        "import streamlit as st\n"
        "import plotly.express as px\n"
        "\n"
        'st.set_page_config(page_title="Housing Price Dashboard", layout="wide")\n'
        "\n"
        'st.title("Housing Price Prediction Dashboard")\n'
        'st.write("Use the sidebar to filter properties and view predictions.")\n'
        "\n"
        "# TD: implement interactive filters and prediction visualisations\n"
    ),
    "src/pipeline/ingest.py": (
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def load_raw_data(path: str) -> pd.DataFrame:\n"
        '    """Load raw housing dataset from CSV."""\n'
        "    return pd.read_csv(Path(path))\n"
    ),
    "src/pipeline/preprocess.py": (
        "import pandas as pd\n"
        "\n"
        "\n"
        "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n"
        '    """Clean data: impute missing values and cap outliers."""\n'
        "    # TD: implement median imputation, IQR outlier capping\n"
        "    return df\n"
    ),
    "src/pipeline/features.py": (
        "import pandas as pd\n"
        "\n"
        "\n"
        "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:\n"
        '    """Create location clusters, price/sqft, temporal features."""\n'
        "    # TD: k-means location clusters, property age, seasonal indicators\n"
        "    return df\n"
    ),
    "src/pipeline/train.py": (
        "import mlflow\n"
        "import optuna  # noqa: F401\n"
        "import xgboost as xgb  # noqa: F401\n"
        "from src.pipeline.ingest import load_raw_data\n"
        "from src.pipeline.preprocess import preprocess\n"
        "from src.pipeline.features import engineer_features\n"
        "\n"
        "\n"
        "def train():\n"
        '    """Train XGBoost model with Optuna tuning and MLflow tracking."""\n'
        '    mlflow.set_experiment("housing-regression")\n'
        "    with mlflow.start_run():\n"
        "        # TD: implement full training pipeline\n"
        "        pass\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    train()\n"
    ),
    "src/pipeline/inference.py": (
        "import mlflow\n"
        "import pandas as pd\n"
        "\n"
        "\n"
        "def predict(features: pd.DataFrame) -> list:\n"
        '    """Run inference using the latest registered model."""\n'
        "    # TD: load model from MLflow registry and run predictions\n"
        "    return []\n"
    ),
    "src/utils/logging.py": (
        "import logging\n"
        "\n"
        "\n"
        "def get_logger(name: str) -> logging.Logger:\n"
        "    logging.basicConfig(\n"
        "        level=logging.INFO,\n"
        '        format="%(asctime)s %(levelname)s %(name)s — %(message)s",\n'
        "    )\n"
        "    return logging.getLogger(name)\n"
    ),
    "src/utils/config.py": (
        "import yaml\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        'def load_config(path: str = "config/config.yaml") -> dict:\n'
        "    with open(Path(path)) as f:\n"
        "        return yaml.safe_load(f)\n"
    ),
    # ── Tests ─────────────────────────────────────────────────────────────────
    "tests/test_api.py": (
        "from fastapi.testclient import TestClient\n"
        "from src.api.main import app\n"
        "\n"
        "client = TestClient(app)\n"
        "\n"
        "\n"
        "def test_health_check():\n"
        '    response = client.get("/health")\n'
        "    assert response.status_code == 200\n"
        '    assert response.json() == {"status": "ok"}\n'
    ),
    "tests/test_pipeline.py": (
        "import pandas as pd\n"
        "from src.pipeline.preprocess import preprocess\n"
        "\n"
        "\n"
        "def test_preprocess_returns_dataframe():\n"
        '    df = pd.DataFrame({"price": [100000, 200000], "bedrooms": [2, 3]})\n'
        "    result = preprocess(df)\n"
        "    assert isinstance(result, pd.DataFrame)\n"
    ),
    "tests/test_features.py": (
        "import pandas as pd\n"
        "from src.pipeline.features import engineer_features\n"
        "\n"
        "\n"
        "def test_engineer_features_returns_dataframe():\n"
        '    df = pd.DataFrame({"price": [100000], "sqft_living": [1500]})\n'
        "    result = engineer_features(df)\n"
        "    assert isinstance(result, pd.DataFrame)\n"
    ),
    # ── gitkeep for empty tracked dirs ────────────────────────────────────────
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "data/features/.gitkeep": "",
    "models/artifacts/.gitkeep": "",
}


def create_structure():
    print("Creating project structure...\n")

    for d in DIRS:
        path = BASE / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [dir]  {d}/")

    print()

    for filename, content in FILES.items():
        path = BASE / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            print(f"  [file] {filename}")
        else:
            print(f"  [skip] {filename} (already exists)")

    print("\n✓ Project structure created successfully.")
    print("\nNext steps:")
    print("  1. uv venv --python 3.11 .venv")
    print("  2. .venv\\Scripts\\Activate.ps1       # Windows PowerShell")
    print("     source .venv/bin/activate          # macOS / Linux")
    print("  3. uv sync")
    print("  4. cp .env.example .env  — then fill in your credentials")


if __name__ == "__main__":
    create_structure()
