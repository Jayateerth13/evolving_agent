"""CLI entry point for the full Orchestrator.

Usage:
    python -m part4_kaggle_orchestrator --demo                  # full pipeline with Titanic
    python -m part4_kaggle_orchestrator --dataset d.csv --target col
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Orchestrator — runs the full multi-agent pipeline",
    )
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--context", type=str, default="experiment_context.json")
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--demo", action="store_true", help="Run with Titanic dataset")
    args = parser.parse_args()

    if args.demo:
        from part2_data_mining.__main__ import download_demo_dataset
        args.dataset = download_demo_dataset()
        args.target = "Survived"

    # Clean slate
    ctx_path = Path(args.context)
    if ctx_path.exists():
        ctx_path.unlink()

    from part4_kaggle_orchestrator.orchestrator.main import run_orchestrator

    run_orchestrator(
        config_path=args.config,
        context_path=args.context,
        dataset_path=args.dataset,
        target_column=args.target,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
