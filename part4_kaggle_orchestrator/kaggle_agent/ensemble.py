"""Ensemble builder — blends top-K models via rank averaging.

Trains each top config, collects OOF predictions, and blends them
with optimized weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict


def build_ensemble(
    df: pd.DataFrame,
    target_column: str,
    top_configs: list[dict[str, Any]],
    task_type: str = "classification",
) -> tuple[np.ndarray, list[float]]:
    """Build a rank-averaged ensemble from top configs.

    Returns (blended_predictions, weights).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    oof_preds: list[np.ndarray] = []

    for cfg in top_configs:
        model_name = cfg.pop("model", "lightgbm")
        params = {k: v for k, v in cfg.items() if k != "model"}
        params["verbose"] = -1 if model_name == "lightgbm" else 0
        params["n_jobs"] = -1

        if model_name == "lightgbm":
            import lightgbm as lgb
            cls = lgb.LGBMClassifier if task_type == "classification" else lgb.LGBMRegressor
        else:
            import xgboost as xgb
            cls = xgb.XGBClassifier if task_type == "classification" else xgb.XGBRegressor

        model = cls(**params)

        if task_type == "classification":
            oof = cross_val_predict(model, X, y, cv=5, method="predict_proba")
            oof_preds.append(oof[:, 1] if oof.ndim > 1 else oof)
        else:
            oof = cross_val_predict(model, X, y, cv=5)
            oof_preds.append(oof)

    n = len(oof_preds)
    weights = [1.0 / n] * n

    ranked = [_rank_normalize(p) for p in oof_preds]
    blended = sum(w * r for w, r in zip(weights, ranked))

    return blended, weights


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Convert predictions to rank-normalized [0, 1] range."""
    from scipy.stats import rankdata
    ranks = rankdata(arr)
    return (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks
