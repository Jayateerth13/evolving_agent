"""Supporting Pydantic models shared across all agents.

These are pure data contracts — no business logic.  Agents import these
to ensure structured communication without coupling to each other.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Feature Engineering ──────────────────────────────────────────────


class FeatureProposal(BaseModel):
    """A proposed feature engineering step returned by the LLM."""

    name: str
    description: str
    code_snippet: str = ""
    rationale: str = ""
    expected_uplift: float = 0.0
    priority: int = 0
    category: str = "statistical"


# ── Model Specification ──────────────────────────────────────────────


class ModelSpec(BaseModel):
    """Specification extracted from a research paper by the Research Copilot."""

    paper_title: str = ""
    paper_id: str = ""
    architecture: str = ""
    framework: str = "sklearn"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: list[str] = Field(default_factory=list)
    training_recipe: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


# ── Dataset Profiling ────────────────────────────────────────────────


class ProfileReport(BaseModel):
    """Structured output from ydata-profiling or equivalent."""

    n_rows: int = 0
    n_cols: int = 0
    target_column: str = ""
    task_type: str = "classification"
    missing_rates: dict[str, float] = Field(default_factory=dict)
    cardinality: dict[str, int] = Field(default_factory=dict)
    dtypes: dict[str, str] = Field(default_factory=dict)
    skew_scores: dict[str, float] = Field(default_factory=dict)
    correlation_with_target: dict[str, float] = Field(default_factory=dict)
    leakage_candidates: list[str] = Field(default_factory=list)
    summary: str = ""


# ── Paper Search ─────────────────────────────────────────────────────


class PaperResult(BaseModel):
    """A research paper returned from Semantic Scholar / ArXiv."""

    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    citation_count: int = 0
    year: int = 0
    has_code: bool = False
    source: str = ""
    relevance_score: float = 0.0


class PaperSections(BaseModel):
    """Extracted text sections from a research paper PDF."""

    abstract: str = ""
    methods: str = ""
    experiments: str = ""
    results: str = ""
    full_text: str = ""


# ── Experiment Comparison ────────────────────────────────────────────


class RunDiff(BaseModel):
    """Structured comparison between two MLflow runs."""

    run_id_a: str
    run_id_b: str
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    feature_importance_shifts: dict[str, float] = Field(default_factory=dict)
    prediction_divergence: float = 0.0
    summary: str = ""


class Hypothesis(BaseModel):
    """A hypothesis for the next experiment iteration."""

    description: str
    rationale: str
    estimated_uplift: float = 0.0
    priority: int = 0
    technique: str = ""


# ── Kaggle ───────────────────────────────────────────────────────────


class CompetitionInfo(BaseModel):
    """Kaggle competition metadata."""

    slug: str
    title: str = ""
    metric: str = ""
    deadline: str = ""
    description: str = ""
    data_description: str = ""
    current_rank: int | None = None
    best_score: float | None = None


class Idea(BaseModel):
    """Idea extracted from Kaggle discussions / notebooks."""

    description: str
    source: str = ""
    category: str = ""
    mention_count: int = 1
    confidence: float = 0.0


class LBScore(BaseModel):
    """Kaggle leaderboard score after submission."""

    public_score: float
    rank: int | None = None
    rank_delta: int | None = None
    total_teams: int | None = None
    submission_id: str = ""


# ── Code Execution ───────────────────────────────────────────────────


class ExecutionResult(BaseModel):
    """Result from running code in the sandbox executor."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    peak_memory_mb: float = 0.0
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    success: bool = True
