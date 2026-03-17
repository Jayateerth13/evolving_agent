"""Lightweight dataset profiling using pandas.

Produces a structured ProfileReport that the LLM uses to propose
feature engineering steps. No heavy dependencies (no ydata-profiling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rdkit_core.models.spec import ProfileReport


def profile_dataset(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
) -> ProfileReport:
    """Profile a tabular dataset and return a structured report."""

    if task_type == "auto":
        n_unique = df[target_column].nunique()
        task_type = "classification" if n_unique <= 20 else "regression"

    missing_rates = (df.isnull().mean()).to_dict()
    missing_rates = {k: round(v, 4) for k, v in missing_rates.items() if v > 0}

    cardinality = {col: int(df[col].nunique()) for col in df.columns}

    dtypes = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            dtypes[col] = "numeric"
        elif pd.api.types.is_bool_dtype(df[col]):
            dtypes[col] = "boolean"
        else:
            dtypes[col] = "categorical"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew_scores = {}
    for col in numeric_cols:
        if col != target_column:
            skew_scores[col] = round(float(df[col].skew()), 4)

    correlation_with_target: dict[str, float] = {}
    if target_column in numeric_cols:
        for col in numeric_cols:
            if col != target_column:
                corr = df[col].corr(df[target_column])
                if pd.notna(corr):
                    correlation_with_target[col] = round(float(corr), 4)

    leakage_candidates = [
        col for col, corr in correlation_with_target.items()
        if abs(corr) > 0.95
    ]

    summary_parts = [
        f"{len(df)} rows, {len(df.columns)} columns",
        f"Target: {target_column} ({task_type})",
        f"{len(missing_rates)} columns with missing values" if missing_rates else "No missing values",
        f"Numeric: {len(numeric_cols)}, Categorical: {sum(1 for v in dtypes.values() if v == 'categorical')}",
    ]

    return ProfileReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        target_column=target_column,
        task_type=task_type,
        missing_rates=missing_rates,
        cardinality=cardinality,
        dtypes=dtypes,
        skew_scores=skew_scores,
        correlation_with_target=correlation_with_target,
        leakage_candidates=leakage_candidates,
        summary=" | ".join(summary_parts),
    )


def format_schema_for_llm(df: pd.DataFrame, target_column: str) -> str:
    """Format dataset schema as a concise string for the LLM prompt."""
    lines = [f"Columns ({len(df.columns)} total):"]
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        null_pct = df[col].isnull().mean() * 100
        sample = df[col].dropna().head(3).tolist()
        marker = " [TARGET]" if col == target_column else ""
        lines.append(
            f"  - {col}{marker}: {dtype}, {n_unique} unique, {null_pct:.1f}% null, sample={sample}"
        )
    return "\n".join(lines)
