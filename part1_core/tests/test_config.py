"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from rdkit_core.config import Settings, load_settings


class TestConfig:
    def test_default_settings(self):
        s = Settings()
        assert s.llm.provider == "nvidia"
        assert s.executor.backend == "local"
        assert s.dataset.task_type == "classification"

    def test_load_from_yaml(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text("""
llm:
  provider: openrouter
  model: nvidia/nemotron-super-120b-a12b
  temperature: 0.5
dataset:
  path: /data/train.csv
  target_column: survived
""")
        s = load_settings(config)
        assert s.llm.provider == "openrouter"
        assert s.llm.temperature == 0.5
        assert s.dataset.target_column == "survived"

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        s = load_settings(tmp_path / "nonexistent.yaml")
        assert s.llm.model == "nvidia/llama-3.3-nemotron-super-49b-v1"

    def test_partial_override(self, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text("""
executor:
  backend: docker
  timeout: 600
""")
        s = load_settings(config)
        assert s.executor.backend == "docker"
        assert s.executor.timeout == 600
        assert s.executor.cpu_limit == 4  # default preserved
