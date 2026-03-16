"""Shared fixtures for Part 1 tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from rdkit_core.models.context import ExperimentContext


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_context() -> ExperimentContext:
    return ExperimentContext(
        run_id="test_run_001",
        iteration=0,
        dataset_version_id="titanic_abc123",
        feature_set_id="feat_v1",
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "fare": [7.25, 71.28, 8.05, 53.1, 8.66],
        "survived": [0, 1, 1, 1, 0],
    })


@pytest.fixture
def context_path(tmp_path: Path, sample_context: ExperimentContext) -> Path:
    path = tmp_path / "experiment_context.json"
    sample_context.save(path)
    return path
