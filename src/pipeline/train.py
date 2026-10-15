"""
Train a baseline XGBoost model.
Tunning is done with Optuna and metrics are logged to MLflow.
Evaluation is done on a new data validation set.

- Reads feature-engineered train/eval CSVs.
- Trains XGBRegressor.
- Performs hyperparameter tuning with Optuna.
- Logs metrics to MLflow.
- Evaluates on new data validation set.
- Returns metrics and saves model to `model_output`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple


import mlflow
import optuna
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DEFAULT_TRAIN = Path("data/features/featured_train_data.csv")
DEFAULT_TEST = Path("data/features/featured_test_data.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")

TARGET = "price"


def _sample(df: pd.DataFrame, sample_data: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_data is None:
        return df
    sample_data = float(sample_data)
    if sample_data <= 0 or sample_data >= 1:
        return df
    return df.sample(frac=sample_data, random_state=random_state).reset_index(drop=True)


def _load_data(
    train_path: Path | str,
    eval_path: Path | str,
    sample_frac: Optional[float],
    random_state: int,
):
    train_df = _sample(pd.read_csv(train_path), sample_frac, random_state)
    eval_df = _sample(pd.read_csv(eval_path), sample_frac, random_state)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_eval, y_eval = eval_df.drop(columns=[TARGET]), eval_df[TARGET]
    return X_train, y_train, X_eval, y_eval


def _score(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_TEST,
    model_output: Path | str = DEFAULT_MODEL,
    model_params: Optional[Dict] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
):
    """Train baseline XGB and save model.

    Returns
    -------
    model : XGBRegressor
    metrics : dict[str, float]
    """
    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path, sample_frac, random_state)

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if model_params:
        params.update(model_params)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    metrics = _score(y_eval, model.predict(X_eval))

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print(f"✅ Model trained. Saved to {out}")
    print(f"   MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.4f}")

    return model, metrics


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_TEST,
    model_output: Path | str = DEFAULT_MODEL,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "xgboost_optuna_housing",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """Run Optuna tuning; save best model; return (best_params, best_metrics)."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path, sample_frac, random_state)

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            metrics = _score(y_eval, model.predict(X_eval))
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        return metrics["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("✅ Best params from Optuna:", best_params)

    # Retrain best model
    best_model = XGBRegressor(
        **{**best_params, "random_state": random_state, "n_jobs": -1, "tree_method": "hist"}
    )
    best_model.fit(X_train, y_train)
    best_metrics = _score(y_eval, best_model.predict(X_eval))
    print("📊 Best tuned model metrics:", best_metrics)

    # Save to models/
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)
    print(f"✅ Best model saved to {out}")

    # Log final best model to MLflow
    with mlflow.start_run(run_name="best_xgb_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.xgboost.log_model(best_model, "model")

    return best_params, best_metrics


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_TEST,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    eval_df = _sample(pd.read_csv(eval_path), sample_frac, random_state)
    X_eval, y_eval = eval_df.drop(columns=[TARGET]), eval_df[TARGET]

    model = load(model_path)
    metrics = _score(y_eval, model.predict(X_eval))

    print("📊 Evaluation:")
    print(f"   MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R²={metrics['r2']:.4f}")
    return metrics


def train(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_TEST,
    model_output: Path | str = DEFAULT_MODEL,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "housing-regression",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """Tune with Optuna, save the best model, then evaluate it.

    Returns (best_params, metrics).
    """
    best_params, _ = tune_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=model_output,
        n_trials=n_trials,
        sample_frac=sample_frac,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        random_state=random_state,
    )

    metrics = evaluate_model(
        model_path=model_output,
        eval_path=eval_path,
        sample_frac=sample_frac,
        random_state=random_state,
    )

    print("✅ Training pipeline complete.")
    return best_params, metrics


if __name__ == "__main__":
    train()
