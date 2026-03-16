"""Configuration loader for the multi-agent system.

Reads config.yaml and exposes typed settings so agents don't need
to parse YAML themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "nvidia"
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    temperature: float = 0.3
    max_tokens: int = 4096


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "rdkit-multiagent"


class ExecutorConfig(BaseModel):
    backend: str = "local"
    timeout: int = 300
    cpu_limit: int = 4
    memory_limit: str = "4g"
    network_disabled: bool = True


class DataStoreConfig(BaseModel):
    base_path: str = "./data/versions"


class DatasetConfig(BaseModel):
    path: str = ""
    target_column: str = ""
    task_type: str = "classification"
    primary_metric: str = "auc"
    metric_target: float = 0.90


class AgentConfig(BaseModel):
    max_iterations: int = 10
    stall_threshold: float = 0.001
    max_stall_count: int = 3
    top_k_features: int = 5
    poll_interval_seconds: int = 10


class Settings(BaseModel):
    """Top-level settings parsed from config.yaml."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    datastore: DataStoreConfig = Field(default_factory=DataStoreConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Load and validate settings from a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        return Settings()

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    raw.pop("project", None)
    return Settings.model_validate(raw)
