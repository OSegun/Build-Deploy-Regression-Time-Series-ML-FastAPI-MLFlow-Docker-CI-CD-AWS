import mlflow
import optuna  # noqa: F401
import xgboost as xgb  # noqa: F401
from src.pipeline.ingest import load_raw_data
from src.pipeline.preprocess import preprocess
from src.pipeline.features import engineer_features


def train():
    """Train XGBoost model with Optuna tuning and MLflow tracking."""
    mlflow.set_experiment("housing-regression")
    with mlflow.start_run():
        # TD: implement full training pipeline
        pass


if __name__ == "__main__":
    train()
