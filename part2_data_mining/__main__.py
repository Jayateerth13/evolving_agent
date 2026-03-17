"""CLI entry point for the Data Mining Agent.

Usage:
    python -m part2_data_mining --dataset path/to/data.csv --target survived
    python -m part2_data_mining --dataset data.csv --target target --config config.yaml
    python -m part2_data_mining --demo  # runs on built-in sample dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_demo_dataset() -> str:
    """Download the Titanic dataset for demo purposes."""
    import urllib.request

    demo_dir = Path("data")
    demo_dir.mkdir(exist_ok=True)
    path = demo_dir / "titanic.csv"

    if path.exists():
        print(f"  Demo dataset already exists: {path}")
        return str(path)

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print(f"  Downloading Titanic dataset...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="part2_data_mining",
        description="Data Mining Agent — autonomous feature engineering and model iteration",
    )
    parser.add_argument("--dataset", type=str, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, help="Target column name")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--context", type=str, default="experiment_context.json", help="Output context path")
    parser.add_argument("--demo", action="store_true", help="Run demo with Titanic dataset")

    args = parser.parse_args()

    if args.demo:
        args.dataset = download_demo_dataset()
        args.target = "Survived"

    if not args.dataset or not args.target:
        parser.error("--dataset and --target are required (or use --demo)")
        sys.exit(1)

    if not Path(args.dataset).exists():
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)

    from part2_data_mining.agent.loop import run_data_mining_agent

    ctx = run_data_mining_agent(
        dataset_path=args.dataset,
        target_column=args.target,
        config_path=args.config,
        context_path=args.context,
    )

    print(f"\n  Final status: {ctx.status.value}")
    print(f"  Iterations: {ctx.iteration + 1}")
    print(f"  Hypotheses logged: {len(ctx.hypothesis_log)}")
    print(f"  Best runs: {ctx.best_run_ids}")
    print()


if __name__ == "__main__":
    main()
