"""Canonical shared boundary contract between all agents.

The ExperimentContext is the single source of truth that lives on disk as
experiment_context.json.  Every agent reads it, mutates its own fields,
and writes it back.  The Orchestrator polls this file to decide which
agent to dispatch next.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Pipeline status — drives the Orchestrator's routing logic."""

    RUNNING = "running"
    STALLED = "stalled"
    STAGED = "staged"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class MetricSet(BaseModel):
    """Metrics for a single evaluation split (train / val / test)."""

    auc: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    rmse: float | None = None
    log_loss: float | None = None
    custom: dict[str, float] = Field(default_factory=dict)


class Metrics(BaseModel):
    """Container holding metric sets for each data split."""

    val: MetricSet = Field(default_factory=MetricSet)
    train: MetricSet | None = None
    test: MetricSet | None = None


class HypothesisEntry(BaseModel):
    """One entry in the hypothesis log written by the Data Mining Agent."""

    iteration: int
    hypothesis: str
    rationale: str
    outcome: str = ""
    metric_before: float = 0.0
    metric_after: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid4().hex[:12]


class ExperimentContext(BaseModel):
    """The shared boundary contract between all agents.

    Schema version is pinned so consumers can validate without importing
    Python — the JSON Schema is exported to ``schemas/experiment_context.json``.
    """

    schema_version: str = "1.0.0"
    run_id: str = Field(default_factory=_short_id)
    iteration: int = 0
    status: Status = Status.RUNNING

    dataset_version_id: str = ""
    feature_set_id: str = ""
    model_spec: dict[str, Any] = Field(default_factory=dict)

    metrics: Metrics = Field(default_factory=Metrics)
    improvement_delta: float = 0.0
    stall_count: int = 0
    max_stall_count: int = 3

    paper_refs: list[str] = Field(default_factory=list)
    hypothesis_log: list[HypothesisEntry] = Field(default_factory=list)
    best_run_ids: list[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    # ── Persistence helpers ──────────────────────────────────────────

    def save(self, path: str | Path = "experiment_context.json") -> Path:
        """Atomically write context to disk (write-tmp then rename)."""
        path = Path(path)
        self.updated_at = _now_iso()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.rename(path)
        return path

    @classmethod
    def load(cls, path: str | Path = "experiment_context.json") -> ExperimentContext:
        """Load context from a JSON file on disk."""
        return cls.model_validate_json(Path(path).read_text())

    # ── Schema export ────────────────────────────────────────────────

    @classmethod
    def export_json_schema(cls, path: str | Path) -> Path:
        """Write the JSON Schema so non-Python consumers can validate."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cls.model_json_schema(), indent=2))
        return path

    # ── Status transition helpers ────────────────────────────────────

    def mark_stalled(self) -> None:
        self.stall_count += 1
        if self.stall_count >= self.max_stall_count:
            self.status = Status.STALLED

    def mark_staged(self) -> None:
        self.status = Status.STAGED

    def mark_promoted(self) -> None:
        self.status = Status.PROMOTED

    def mark_rejected(self) -> None:
        self.status = Status.REJECTED

    def reset_for_new_iteration(self) -> None:
        """Called by the Research Copilot after injecting a new model."""
        self.status = Status.RUNNING
        self.stall_count = 0
        self.iteration += 1

    def record_hypothesis(
        self,
        hypothesis: str,
        rationale: str,
        metric_before: float = 0.0,
        metric_after: float = 0.0,
        outcome: str = "",
    ) -> None:
        self.hypothesis_log.append(
            HypothesisEntry(
                iteration=self.iteration,
                hypothesis=hypothesis,
                rationale=rationale,
                outcome=outcome,
                metric_before=metric_before,
                metric_after=metric_after,
            )
        )
