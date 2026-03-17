"""AutoML sweep over XGBoost, LightGBM, and CatBoost using Optuna.

Logs every trial to MLflow. Returns the best config and top-K run IDs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from rdkit_core.tools.tracker import ExperimentTracker


def run_automl_sweep(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    n_trials: int = 20,
    tracker: ExperimentTracker | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Run an Optuna sweep. Returns (best_config, top_run_ids)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scoring = "roc_auc" if task_type == "classification" else "neg_root_mean_squared_error"
    trials_log: list[tuple[float, dict[str, Any], str | None]] = []

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", ["lightgbm", "xgboost"])

        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        if model_name == "lightgbm":
            import lightgbm as lgb
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
            params["verbose"] = -1
            params["n_jobs"] = -1
            cls = lgb.LGBMClassifier if task_type == "classification" else lgb.LGBMRegressor
        else:
            import xgboost as xgb
            params["verbosity"] = 0
            params["n_jobs"] = -1
            cls = xgb.XGBClassifier if task_type == "classification" else xgb.XGBRegressor

        model = cls(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        mean_score = float(np.mean(scores))

        run_id = None
        if tracker:
            metric_key = "val_auc" if task_type == "classification" else "val_rmse"
            metric_val = mean_score if task_type == "classification" else -mean_score
            run_id = tracker.log_run(
                params={"model": model_name, **params},
                metrics={metric_key: metric_val, f"{metric_key}_std": float(np.std(scores))},
                run_name=f"sweep_trial_{trial.number}",
            )

        trials_log.append((mean_score, {"model": model_name, **params}, run_id))
        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_log.sort(key=lambda t: t[0], reverse=True)
    best_config = study.best_params
    top_run_ids = [t[2] for t in trials_log[:5] if t[2] is not None]

    return best_config, top_run_ids
