"""Kaggle Agent — FE catalog + AutoML sweep + ensemble + promotion.

Reads experiment_context.json (status must be 'staged'), runs
aggressive feature engineering, an Optuna sweep, ensembles the top
models, and promotes or rejects the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rdkit_core import ExperimentContext, Status
from rdkit_core.config import load_settings
from rdkit_core.tools.datastore import DataStore
from rdkit_core.tools.tracker import ExperimentTracker

from .ensemble import build_ensemble
from .fe_catalog import apply_catalog
from .sweep import run_automl_sweep

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _header(text: str) -> None:
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}{RESET}")


def _step(icon: str, text: str) -> None:
    print(f"  {icon} {text}")


def run_kaggle_agent(
    context_path: str = "experiment_context.json",
    config_path: str = "config.yaml",
    n_trials: int = 15,
    target_override: str = "",
) -> ExperimentContext:
    """Run the Kaggle Agent pipeline."""
    settings = load_settings(config_path)
    ctx = ExperimentContext.load(context_path)

    if ctx.status != Status.STAGED:
        _step("⏭️", f"Status is '{ctx.status.value}', not 'staged' — nothing to do")
        return ctx

    _header("Kaggle Agent — Activating")
    _step("📖", f"Context: iteration={ctx.iteration}, best_auc={ctx.metrics.val.auc:.4f}")

    target = target_override or settings.dataset.target_column
    task_type = settings.dataset.task_type
    primary_metric = settings.dataset.primary_metric
    baseline_metric = ctx.metrics.val.auc

    datastore = DataStore(base_path=settings.datastore.base_path)
    try:
        tracker = ExperimentTracker(
            tracking_uri=settings.mlflow.tracking_uri,
            experiment_name=settings.mlflow.experiment_name,
        )
    except Exception:
        tracker = None  # type: ignore[assignment]

    # ── Step 1: Load dataset + FE catalog ────────────────────────
    _header("Step 1 — Feature Engineering Catalog")

    if ctx.dataset_version_id and datastore.version_exists(ctx.dataset_version_id):
        df = datastore.load_dataset(ctx.dataset_version_id)
        _step("📂", f"Loaded dataset: {ctx.dataset_version_id}")
    else:
        csv_path = settings.dataset.path or "data/titanic.csv"
        df = pd.read_csv(csv_path)
        _step("📂", f"Loaded from CSV: {csv_path}")

    _step("⚙️", "Applying full FE catalog...")
    df_fe = apply_catalog(df, target=target)
    new_cols = len(df_fe.columns) - len(df.columns)
    _step("✅", f"{len(df_fe.columns)} features ({new_cols} new from catalog)")

    # ── Step 2: AutoML sweep ─────────────────────────────────────
    _header("Step 2 — AutoML Sweep (Optuna)")
    _step("🔄", f"Running {n_trials} trials over LightGBM + XGBoost...")

    best_config, top_run_ids = run_automl_sweep(
        df=df_fe,
        target_column=target,
        task_type=task_type,
        n_trials=n_trials,
        tracker=tracker,
    )

    _step("🏆", f"Best config: {best_config}")
    _step("📊", f"Top {len(top_run_ids)} runs logged to MLflow")

    # ── Step 3: Ensemble ─────────────────────────────────────────
    _header("Step 3 — Ensemble")

    top_configs = []
    if tracker and top_run_ids:
        for rid in top_run_ids[:3]:
            try:
                run_data = tracker.get_run(rid)
                cfg = {k: _try_numeric(v) for k, v in run_data["params"].items()}
                top_configs.append(cfg)
            except Exception:
                pass

    if not top_configs:
        top_configs = [{"model": best_config.get("model", "lightgbm"), **best_config}]

    _step("🔗", f"Blending {len(top_configs)} models via rank averaging...")

    blended, weights = build_ensemble(
        df=df_fe,
        target_column=target,
        top_configs=top_configs,
        task_type=task_type,
    )

    if task_type == "classification":
        ensemble_auc = roc_auc_score(df_fe[target], blended)
        _step(CYAN + "📈", f"Ensemble AUC: {ensemble_auc:.4f}{RESET}")
        improvement = ensemble_auc - baseline_metric
    else:
        ensemble_auc = 0.0
        improvement = 0.0

    # ── Step 4: Promote or reject ────────────────────────────────
    _header("Step 4 — Final Decision")

    if improvement > 0:
        ctx.status = Status.PROMOTED
        ctx.metrics.val.auc = ensemble_auc
        ctx.improvement_delta = improvement
        _step(GREEN + "🎉", f"PROMOTED — AUC improved by {improvement:.4f}{RESET}")
        _step("📈", f"  Baseline: {baseline_metric:.4f} → Ensemble: {ensemble_auc:.4f}")
    else:
        ctx.status = Status.REJECTED
        _step(RED + "❌", f"REJECTED — no improvement over baseline{RESET}")

    if top_run_ids:
        ctx.best_run_ids.extend(top_run_ids[:3])

    ctx.save(context_path)
    _step("💾", f"Context written: status={ctx.status.value}")

    return ctx


def _try_numeric(v: str) -> object:
    """Convert MLflow string params back to numeric where possible."""
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        return v
