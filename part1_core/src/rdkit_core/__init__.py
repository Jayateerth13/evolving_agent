"""rdkit-core: Shared contracts and tool layer for the R&D Multi-Agent ML System.

This package is the foundation that all agents (Parts 2–4) install via pip.
It provides:
  - Pydantic data models (the shared contract)
  - Sandboxed code execution (Docker or local subprocess)
  - MLflow experiment tracking and model registry
  - Content-addressed versioned dataset storage
  - Unified LLM client for NVIDIA Nemotron

Usage::

    from rdkit_core import ExperimentContext, Status
    from rdkit_core import CodeExecutor, ExperimentTracker, DataStore, ModelRegistry, LLMClient
    from rdkit_core import ModelSpec, FeatureProposal, ProfileReport, RunDiff
"""

__version__ = "0.1.0"

# ── Models (shared contracts) ────────────────────────────────────────
from rdkit_core.models.context import (
    ExperimentContext,
    HypothesisEntry,
    MetricSet,
    Metrics,
    Status,
)
from rdkit_core.models.spec import (
    CompetitionInfo,
    ExecutionResult,
    FeatureProposal,
    Hypothesis,
    Idea,
    LBScore,
    ModelSpec,
    PaperResult,
    PaperSections,
    ProfileReport,
    RunDiff,
)

# ── Tools ────────────────────────────────────────────────────────────
from rdkit_core.tools.datastore import DataStore
from rdkit_core.tools.executor import CodeExecutor, create_executor
from rdkit_core.tools.llm_client import LLMClient
from rdkit_core.tools.registry import ModelRegistry
from rdkit_core.tools.tracker import ExperimentTracker

__all__ = [
    # Models
    "CompetitionInfo",
    "ExecutionResult",
    "ExperimentContext",
    "FeatureProposal",
    "Hypothesis",
    "HypothesisEntry",
    "Idea",
    "LBScore",
    "MetricSet",
    "Metrics",
    "ModelSpec",
    "PaperResult",
    "PaperSections",
    "ProfileReport",
    "RunDiff",
    "Status",
    # Tools
    "CodeExecutor",
    "DataStore",
    "ExperimentTracker",
    "LLMClient",
    "ModelRegistry",
    "create_executor",
]
