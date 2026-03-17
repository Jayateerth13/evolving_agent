"""CLI entry point for the Kaggle Agent (standalone).

Usage:
    python -m part4_kaggle_orchestrator.kaggle_agent --context experiment_context.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="kaggle_agent")
    parser.add_argument("--context", default="experiment_context.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", default="Survived")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    if not Path(args.context).exists():
        print(f"Error: {args.context} not found")
        sys.exit(1)

    from part4_kaggle_orchestrator.kaggle_agent.agent import run_kaggle_agent

    ctx = run_kaggle_agent(
        context_path=args.context,
        config_path=args.config,
        n_trials=args.trials,
        target_override=args.target,
    )
    print(f"\n  Final status: {ctx.status.value}")
    print(f"  AUC: {ctx.metrics.val.auc:.4f}")
    print()


if __name__ == "__main__":
    main()
