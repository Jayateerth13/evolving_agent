"""Feature engineering catalog — composable transforms for tabular data.

Each transform is a function (DataFrame, config) -> DataFrame.
The Kaggle Agent chains these based on the dataset profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fill_missing_numeric(df: pd.DataFrame, **kw: object) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    return df.fillna({c: df[c].median() for c in num})


def fill_missing_categorical(df: pd.DataFrame, **kw: object) -> pd.DataFrame:
    cat = df.select_dtypes(include=["object", "category"]).columns
    return df.fillna({c: df[c].mode().iloc[0] if not df[c].mode().empty else "MISSING" for c in cat})


def label_encode(df: pd.DataFrame, **kw: object) -> pd.DataFrame:
    result = df.copy()
    for c in result.select_dtypes(include=["object"]).columns:
        result[c] = result[c].astype("category").cat.codes
    return result


def frequency_encode(df: pd.DataFrame, **kw: object) -> pd.DataFrame:
    result = df.copy()
    for c in result.select_dtypes(include=["object", "category"]).columns:
        freq = result[c].value_counts(normalize=True)
        result[f"{c}_freq"] = result[c].map(freq).astype(float)
    return result


def add_interactions(df: pd.DataFrame, target: str = "", **kw: object) -> pd.DataFrame:
    """Add pairwise multiplication of top numeric features."""
    result = df.copy()
    num_cols = [c for c in result.select_dtypes(include=[np.number]).columns if c != target][:5]
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            result[f"{a}_x_{b}"] = result[a] * result[b]
    return result


def log_transform_skewed(df: pd.DataFrame, threshold: float = 1.0, **kw: object) -> pd.DataFrame:
    result = df.copy()
    for c in result.select_dtypes(include=[np.number]).columns:
        if result[c].skew() > threshold:
            result[f"{c}_log"] = np.log1p(result[c].clip(lower=0))
    return result


CATALOG = {
    "fill_missing_numeric": fill_missing_numeric,
    "fill_missing_categorical": fill_missing_categorical,
    "label_encode": label_encode,
    "frequency_encode": frequency_encode,
    "add_interactions": add_interactions,
    "log_transform_skewed": log_transform_skewed,
}


def apply_catalog(
    df: pd.DataFrame,
    steps: list[str] | None = None,
    target: str = "",
) -> pd.DataFrame:
    """Apply a sequence of catalog transforms. Defaults to all."""
    chosen = steps or list(CATALOG.keys())
    result = df.copy()
    for name in chosen:
        fn = CATALOG.get(name)
        if fn:
            result = fn(result, target=target)
    return result
