"""Orchestrator -- the system entry point.

Polls experiment_context.json and dispatches the correct agent
as a subprocess based on the status field. Never imports agent
source code -- communicates exclusively through the JSON file
and subprocess calls.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from rdkit_core import ExperimentContext, Status
from rdkit_core.config import load_settings

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _banner(text: str) -> None:
    width = 60
    print(f"\n{BOLD}{'=' * width}")
    print(f"  ORCHESTRATOR -- {text}")
    print(f"{'=' * width}{RESET}")


def _log(text: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  {DIM}[{ts}]{RESET} {text}")


def _run_agent(cmd: list[str], label: str) -> int:
    """Spawn an agent as a subprocess and stream its output."""
    _log(f"Spawning {label}: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        env=None,
    )
    return proc.returncode


def _print_summary(ctx: ExperimentContext) -> None:
    """Print a final summary of the pipeline run."""
    _log(f"Final AUC: {ctx.metrics.val.auc:.4f}")
    _log(f"Total iterations: {ctx.iteration}")
    _log(f"Best runs: {ctx.best_run_ids}")
    _log(f"Paper refs: {ctx.paper_refs}")
    _log(f"Hypotheses: {len(ctx.hypothesis_log)}")


def run_orchestrator(
    config_path: str = "config.yaml",
    context_path: str = "experiment_context.json",
    dataset_path: str = "",
    target_column: str = "",
    max_cycles: int = 4,
) -> None:
    """Run the full orchestration loop."""
    settings = load_settings(config_path)
    python = sys.executable

    _banner("Starting Pipeline")
    _log(f"Config: {config_path}")
    _log(f"Context: {context_path}")
    _log(f"Max cycles: {max_cycles}")

    if not Path(context_path).exists():
        if not dataset_path:
            dataset_path = settings.dataset.path or "data/titanic.csv"
        if not target_column:
            target_column = settings.dataset.target_column or "Survived"

        _log(f"No context file -- bootstrapping with dataset={dataset_path}")
        ctx = ExperimentContext()
        ctx.save(context_path)

    for cycle in range(1, max_cycles + 1):
        ctx = ExperimentContext.load(context_path)
        status = ctx.status

        _banner(f"Cycle {cycle}/{max_cycles} -- status: {status.value}")

        if status == Status.RUNNING:
            ds = dataset_path or settings.dataset.path or "data/titanic.csv"
            tgt = target_column or settings.dataset.target_column or "Survived"
            rc = _run_agent(
                [python, "-m", "part2_data_mining",
                 "--dataset", ds, "--target", tgt,
                 "--config", config_path, "--context", context_path],
                "Data Mining Agent (Part 2)",
            )
            if rc != 0:
                _log(f"{YELLOW}Part 2 exited with code {rc}{RESET}")

        elif status == Status.STALLED:
            rc = _run_agent(
                [python, "-m", "part3_research_copilot",
                 "--context", context_path, "--config", config_path],
                "Research Copilot (Part 3)",
            )
            if rc != 0:
                _log(f"{YELLOW}Part 3 exited with code {rc}{RESET}")

        elif status == Status.STAGED:
            tgt = target_column or settings.dataset.target_column or "Survived"
            rc = _run_agent(
                [python, "-m", "part4_kaggle_orchestrator.kaggle_agent",
                 "--context", context_path, "--config", config_path,
                 "--target", tgt],
                "Kaggle Agent (Part 4)",
            )
            if rc != 0:
                _log(f"{YELLOW}Part 4 exited with code {rc}{RESET}")

        elif status in (Status.PROMOTED, Status.REJECTED):
            _banner(f"Pipeline Complete -- {status.value.upper()}")
            _print_summary(ctx)
            return

        # Check if agent just moved us to a terminal state
        ctx_after = ExperimentContext.load(context_path)
        if ctx_after.status in (Status.PROMOTED, Status.REJECTED):
            _banner(f"Pipeline Complete -- {ctx_after.status.value.upper()}")
            _print_summary(ctx_after)
            return

        time.sleep(1)

    _banner("Max cycles reached")
    ctx_final = ExperimentContext.load(context_path)
    _print_summary(ctx_final)
