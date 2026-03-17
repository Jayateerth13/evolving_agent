"""CLI entry point for the Research Copilot Agent.

Usage:
    python -m part3_research_copilot --context experiment_context.json
    python -m part3_research_copilot  # defaults to experiment_context.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="part3_research_copilot",
        description="Research Copilot — finds papers, generates model code when pipeline stalls",
    )
    parser.add_argument(
        "--context", type=str, default="experiment_context.json",
        help="Path to experiment_context.json",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    if not Path(args.context).exists():
        print(f"Error: context file not found: {args.context}")
        sys.exit(1)

    from part3_research_copilot.agent.loop import run_research_copilot

    ctx = run_research_copilot(
        context_path=args.context,
        config_path=args.config,
    )

    print(f"\n  Final status: {ctx.status.value}")
    print(f"  Iteration: {ctx.iteration}")
    print(f"  Paper refs: {ctx.paper_refs}")
    print(f"  Model spec: {ctx.model_spec.get('architecture', 'N/A')}")
    print()


if __name__ == "__main__":
    main()
