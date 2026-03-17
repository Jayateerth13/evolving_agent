"""Main iteration loop — the core of the Data Mining Agent.

Runs the cycle: profile → propose features → train → compare → hypothesize → repeat.
Exits by writing experiment_context.json with status 'stalled' or 'staged'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rdkit_core import ExperimentContext, Status
from rdkit_core.config import load_settings
from rdkit_core.models.spec import FeatureProposal, RunDiff
from rdkit_core.tools.datastore import DataStore
from rdkit_core.tools.llm_client import LLMClient
from rdkit_core.tools.tracker import ExperimentTracker

from .profiler import format_schema_for_llm, profile_dataset
from .proposer import propose_features, propose_next_step
from .trainer import run_experiment

# ── Pretty console output ────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _header(text: str) -> None:
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}{RESET}")


def _step(icon: str, text: str) -> None:
    print(f"  {icon} {text}")


def _metric(name: str, value: float, delta: float | None = None) -> None:
    delta_str = ""
    if delta is not None:
        color = GREEN if delta > 0 else (RED if delta < 0 else DIM)
        sign = "+" if delta > 0 else ""
        delta_str = f"  {color}({sign}{delta:.4f}){RESET}"
    print(f"    {CYAN}{name}: {value:.4f}{RESET}{delta_str}")


# ── Main loop ─────────────────────────────────────────────────────────


def run_data_mining_agent(
    dataset_path: str,
    target_column: str,
    config_path: str = "config.yaml",
    context_path: str = "experiment_context.json",
) -> ExperimentContext:
    """Run the full data mining loop.

    This is the entry point called by __main__.py or the Orchestrator.
    """
    settings = load_settings(config_path)

    task_type = settings.dataset.task_type
    primary_metric = settings.dataset.primary_metric
    metric_target = settings.dataset.metric_target
    max_iterations = settings.agent.max_iterations
    stall_threshold = settings.agent.stall_threshold

    _header("Data Mining Agent — Initializing")

    # ── Setup tools ──────────────────────────────────────────────
    llm = LLMClient(
        model=settings.llm.model,
        provider=settings.llm.provider,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )
    try:
        tracker = ExperimentTracker(
            tracking_uri=settings.mlflow.tracking_uri,
            experiment_name=settings.mlflow.experiment_name,
        )
    except Exception:
        _step("⚠️ ", "MLflow unavailable — running without experiment tracking")
        tracker = None  # type: ignore[assignment]
    datastore = DataStore(base_path=settings.datastore.base_path)

    # ── Load + profile dataset ───────────────────────────────────
    _step("📂", f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    _step("📊", f"Profiling: {len(df)} rows, {len(df.columns)} columns")

    profile = profile_dataset(df, target_column, task_type)
    schema_str = format_schema_for_llm(df, target_column)

    version_id = datastore.save_dataset(df, name=Path(dataset_path).stem)
    _step("💾", f"Dataset saved: {version_id}")

    if profile.missing_rates:
        missing = ", ".join(f"{k} ({v*100:.1f}%)" for k, v in profile.missing_rates.items())
        _step("⚠️ ", f"Missing: {missing}")
    if profile.leakage_candidates:
        _step("🚨", f"Leakage risk: {profile.leakage_candidates}")

    # ── Initialize context ───────────────────────────────────────
    ctx = ExperimentContext(
        dataset_version_id=version_id,
        max_stall_count=settings.agent.max_stall_count,
    )

    best_metric = 0.0
    best_run_id: str | None = None
    all_proposals: list[FeatureProposal] = []

    # ── Iteration loop ───────────────────────────────────────────
    for iteration in range(max_iterations):
        ctx.iteration = iteration
        is_baseline = iteration == 0

        _header(f"Iteration {iteration}" + (" — Baseline" if is_baseline else ""))

        # ── Step 1: Propose features via LLM ─────────────────────
        _step("🤖", "Asking Nemotron for feature engineering ideas...")

        if is_baseline:
            proposals = propose_features(profile, schema_str, llm, top_k=settings.agent.top_k_features)
        else:
            diff = _compute_diff(tracker, best_run_id, ctx)
            hypothesis = propose_next_step(
                diff,
                [h.model_dump() for h in ctx.hypothesis_log],
                profile,
                llm,
            )
            _step("💡", f"Hypothesis: {hypothesis.description}")
            proposals = propose_features(
                profile, schema_str, llm,
                strategy=hypothesis.technique,
                top_k=3,
            )

        for p in proposals:
            _step("  →", f"{p.name}: {p.description} {DIM}(+{p.expected_uplift:.3f}){RESET}")
        all_proposals = proposals if is_baseline else all_proposals + proposals

        # ── Step 2: Train with proposed features ─────────────────
        _step("🏋️", "Training LightGBM (5-fold CV)...")

        run_id, metrics, applied = run_experiment(
            df=df,
            target_column=target_column,
            task_type=task_type,
            proposals=all_proposals,
            tracker=tracker,
            iteration=iteration,
        )

        current_metric = metrics.get(f"val_{primary_metric}", 0.0)
        delta = current_metric - best_metric if not is_baseline else None

        for name, val in metrics.items():
            if not name.endswith("_std"):
                _metric(name, val, delta if name == f"val_{primary_metric}" else None)

        if applied:
            _step("✅", f"Applied: {', '.join(applied)}")

        # ── Step 3: Compare + update state ───────────────────────
        if is_baseline or current_metric > best_metric:
            improvement = current_metric - best_metric
            best_metric = current_metric
            best_run_id = run_id
            if run_id:
                ctx.best_run_ids.append(run_id)
            ctx.improvement_delta = improvement

            if not is_baseline:
                _step(GREEN + "📈", f"New best! {primary_metric} = {best_metric:.4f}{RESET}")
        else:
            ctx.mark_stalled()
            _step(YELLOW + "📉", f"No improvement (delta={delta:.4f}), stall {ctx.stall_count}/{ctx.max_stall_count}{RESET}")

        ctx.metrics.val.auc = best_metric
        ctx.feature_set_id = version_id

        ctx.record_hypothesis(
            hypothesis=proposals[0].description if proposals else "baseline",
            rationale=proposals[0].rationale if proposals else "initial run",
            metric_before=best_metric - (delta or 0),
            metric_after=current_metric,
            outcome="improved" if (delta or 0) > stall_threshold else "no_improvement",
        )

        # ── Step 4: Check termination ────────────────────────────
        if ctx.status == Status.STALLED:
            _header(f"STALLED after {iteration + 1} iterations")
            _step("📝", f"Best {primary_metric}: {best_metric:.4f}")
            _step("🔄", "Research Copilot will take over...")
            break

        if best_metric >= metric_target:
            ctx.mark_staged()
            _header(f"TARGET REACHED — {primary_metric} = {best_metric:.4f}")
            _step("🎯", f"Target was {metric_target:.4f}")
            _step("➡️ ", "Handing off to Kaggle Agent...")
            break

    # ── Write context + exit ─────────────────────────────────────
    if ctx.status == Status.RUNNING:
        ctx.mark_staged()
        _header(f"Max iterations reached — best {primary_metric}: {best_metric:.4f}")

    ctx.save(context_path)
    _step("💾", f"Context written to {context_path} (status: {ctx.status.value})")
    return ctx


def _compute_diff(
    tracker: ExperimentTracker,
    best_run_id: str | None,
    ctx: ExperimentContext,
) -> RunDiff | None:
    """Try to compute a run diff; return None if MLflow isn't available."""
    if not best_run_id or len(ctx.best_run_ids) < 2:
        return None
    try:
        return tracker.diff_runs(ctx.best_run_ids[-2], ctx.best_run_ids[-1])
    except Exception:
        return None
