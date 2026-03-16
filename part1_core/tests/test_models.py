"""Tests for the Pydantic data models (shared contracts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdkit_core.models.context import (
    ExperimentContext,
    HypothesisEntry,
    MetricSet,
    Metrics,
    Status,
)
from rdkit_core.models.spec import (
    ExecutionResult,
    FeatureProposal,
    ModelSpec,
    ProfileReport,
    RunDiff,
)


class TestStatus:
    def test_enum_values(self):
        assert Status.RUNNING == "running"
        assert Status.STALLED == "stalled"
        assert Status.STAGED == "staged"
        assert Status.PROMOTED == "promoted"
        assert Status.REJECTED == "rejected"

    def test_from_string(self):
        assert Status("running") is Status.RUNNING


class TestExperimentContext:
    def test_defaults(self):
        ctx = ExperimentContext()
        assert ctx.status == Status.RUNNING
        assert ctx.iteration == 0
        assert ctx.stall_count == 0
        assert ctx.improvement_delta == 0.0
        assert len(ctx.run_id) == 12
        assert ctx.schema_version == "1.0.0"

    def test_save_and_load(self, tmp_path: Path):
        ctx = ExperimentContext(run_id="abc123")
        path = tmp_path / "ctx.json"
        ctx.save(path)

        loaded = ExperimentContext.load(path)
        assert loaded.run_id == "abc123"
        assert loaded.status == Status.RUNNING

    def test_atomic_save(self, tmp_path: Path):
        """Save should not leave a half-written file on crash."""
        ctx = ExperimentContext(run_id="atomic_test")
        path = tmp_path / "ctx.json"
        ctx.save(path)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()

    def test_json_roundtrip(self):
        ctx = ExperimentContext(run_id="roundtrip")
        raw = ctx.model_dump_json()
        restored = ExperimentContext.model_validate_json(raw)
        assert restored.run_id == "roundtrip"

    def test_mark_stalled(self):
        ctx = ExperimentContext(max_stall_count=3)
        ctx.mark_stalled()
        assert ctx.stall_count == 1
        assert ctx.status == Status.RUNNING  # not stalled yet

        ctx.mark_stalled()
        ctx.mark_stalled()
        assert ctx.stall_count == 3
        assert ctx.status == Status.STALLED

    def test_mark_staged(self):
        ctx = ExperimentContext()
        ctx.mark_staged()
        assert ctx.status == Status.STAGED

    def test_reset_for_new_iteration(self):
        ctx = ExperimentContext(stall_count=3, status=Status.STALLED, iteration=2)
        ctx.reset_for_new_iteration()
        assert ctx.status == Status.RUNNING
        assert ctx.stall_count == 0
        assert ctx.iteration == 3

    def test_record_hypothesis(self):
        ctx = ExperimentContext()
        ctx.record_hypothesis(
            hypothesis="Add polynomial features",
            rationale="Feature interactions may help",
            metric_before=0.75,
            metric_after=0.78,
            outcome="improved",
        )
        assert len(ctx.hypothesis_log) == 1
        assert ctx.hypothesis_log[0].hypothesis == "Add polynomial features"
        assert ctx.hypothesis_log[0].iteration == 0

    def test_export_json_schema(self, tmp_path: Path):
        schema_path = tmp_path / "schema.json"
        ExperimentContext.export_json_schema(schema_path)
        assert schema_path.exists()

        schema = json.loads(schema_path.read_text())
        assert "properties" in schema
        assert "run_id" in schema["properties"]
        assert "status" in schema["properties"]


class TestMetrics:
    def test_default_metric_set(self):
        ms = MetricSet()
        assert ms.auc == 0.0
        assert ms.rmse is None

    def test_custom_metrics(self):
        ms = MetricSet(auc=0.85, custom={"mape": 0.12})
        assert ms.custom["mape"] == 0.12

    def test_metrics_container(self):
        m = Metrics(val=MetricSet(auc=0.9))
        assert m.val.auc == 0.9
        assert m.train is None


class TestSpecModels:
    def test_feature_proposal(self):
        fp = FeatureProposal(
            name="age_bins",
            description="Bin age into categories",
            expected_uplift=0.02,
        )
        assert fp.category == "statistical"

    def test_model_spec(self):
        spec = ModelSpec(
            paper_title="TabNet",
            architecture="attention-based",
            hyperparameters={"n_steps": 3, "n_a": 8},
        )
        assert spec.framework == "sklearn"

    def test_profile_report(self):
        pr = ProfileReport(
            n_rows=1000,
            n_cols=10,
            target_column="survived",
            missing_rates={"age": 0.2},
        )
        assert pr.task_type == "classification"

    def test_run_diff(self):
        diff = RunDiff(
            run_id_a="run1",
            run_id_b="run2",
            metric_deltas={"auc": 0.03},
        )
        assert diff.metric_deltas["auc"] == 0.03

    def test_execution_result(self):
        er = ExecutionResult(stdout="ok", exit_code=0, success=True)
        assert er.timed_out is False
