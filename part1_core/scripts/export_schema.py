#!/usr/bin/env python3
"""Export JSON Schema for ExperimentContext to schemas/ directory."""

from pathlib import Path

from rdkit_core.models.context import ExperimentContext


def main() -> None:
    schema_dir = Path(__file__).resolve().parent.parent / "schemas"
    out = ExperimentContext.export_json_schema(schema_dir / "experiment_context.json")
    print(f"Exported JSON Schema to {out}")


if __name__ == "__main__":
    main()
