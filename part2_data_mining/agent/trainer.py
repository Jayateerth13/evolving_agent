"""LightGBM training wrapper with cross-validation and MLflow logging.

Handles both classification and regression. Designed to be fast for
demo purposes (LightGBM trains in <1s on small datasets).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from rdkit_core.models.spec import FeatureProposal
from rdkit_core.tools.tracker import ExperimentTracker


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    proposals: list[FeatureProposal] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply feature proposals and prepare X for training.

    Returns (df_processed, feature_columns, applied_names).
    """
    result = df.copy()
    applied: list[str] = []

    if proposals:
        for p in proposals:
            if not p.code_snippet.strip():
                continue
            try:
                exec(p.code_snippet, {"df": result, "pd": pd, "np": np})
                applied.append(p.name)
            except Exception as e:
                print(f"    [skip] {p.name}: {e}")

    cat_cols = result.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if col != target_column:
            result[col] = result[col].astype("category").cat.codes

    drop_cols = [target_column]
    drop_cols += [c for c in result.columns if result[c].dtype == "object"]
    feature_cols = [c for c in result.columns if c not in drop_cols]

    return result, feature_cols, applied


def run_experiment(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    proposals: list[FeatureProposal] | None = None,
    model_params: dict[str, Any] | None = None,
    tracker: ExperimentTracker | None = None,
    iteration: int = 0,
    parent_run_id: str | None = None,
) -> tuple[str | None, dict[str, float], list[str]]:
    """Train a LightGBM model with cross-validation.

    Returns (run_id, metrics_dict, applied_feature_names).
    """
    import lightgbm as lgb

    result, feature_cols, applied = prepare_features(df, target_column, proposals)

    X = result[feature_cols]
    y = result[target_column]

    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
    }
    params = {**defaults, **(model_params or {})}

    if task_type == "classification":
        model = lgb.LGBMClassifier(**params)
        scoring = "roc_auc"
    else:
        model = lgb.LGBMRegressor(**params)
        scoring = "neg_root_mean_squared_error"

    cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)

    if task_type == "classification":
        metrics = {
            "val_auc": round(float(np.mean(cv_scores)), 6),
            "val_auc_std": round(float(np.std(cv_scores)), 6),
        }
    else:
        metrics = {
            "val_rmse": round(float(-np.mean(cv_scores)), 6),
            "val_rmse_std": round(float(np.std(cv_scores)), 6),
        }

    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    top_features = sorted(importances, key=importances.get, reverse=True)[:10]  # type: ignore[arg-type]
    fi_metrics = {f"fi_{f}": round(importances[f], 4) for f in top_features}

    run_id = None
    if tracker:
        log_params: dict[str, Any] = {
            "iteration": iteration,
            "model": "LightGBM",
            "task_type": task_type,
            "n_features": len(feature_cols),
            "features_applied": applied,
            **{f"lgb_{k}": v for k, v in params.items()},
        }
        run_id = tracker.log_run(
            params=log_params,
            metrics={**metrics, **fi_metrics},
            run_name=f"iter_{iteration}",
            parent_run_id=parent_run_id,
        )

    return run_id, metrics, applied
