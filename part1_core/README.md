# rdkit-core

Foundation package for the R&D Multi-Agent ML System. All other parts (2–4) install this as a pip dependency.

## Install

```bash
pip install -e ".[dev]"
```

## What's Inside

### Models (Shared Contracts)

```python
from rdkit_core import ExperimentContext, Status, ModelSpec, FeatureProposal, ProfileReport
```

| Model | Purpose |
|-------|---------|
| `ExperimentContext` | The boundary contract — agents read/write this JSON file |
| `ModelSpec` | Model specification extracted from papers (Research Copilot) |
| `FeatureProposal` | Feature engineering proposals (Data Mining Agent) |
| `ProfileReport` | Dataset profiling output |
| `RunDiff` | Comparison between two experiment runs |
| `PaperResult` | Research paper search result |
| `CompetitionInfo` | Kaggle competition metadata |

### Tools

```python
from rdkit_core import CodeExecutor, ExperimentTracker, DataStore, ModelRegistry, LLMClient
```

| Tool | Purpose |
|------|---------|
| `CodeExecutor` / `create_executor()` | Run untrusted Python in a sandbox (Docker or local) |
| `ExperimentTracker` | MLflow wrapper — log runs, compare runs, list best |
| `DataStore` | Content-hashed versioned dataset storage |
| `ModelRegistry` | MLflow model registry — register, promote, get best |
| `LLMClient` | Unified NVIDIA Nemotron client (chat, JSON, structured, tools) |

### Configuration

```python
from rdkit_core.config import load_settings

settings = load_settings("config.yaml")
```

## For Other Parts

Install from the repo root:

```bash
pip install -e ./part1_core
```

Then import what you need:

```python
from rdkit_core import ExperimentContext, create_executor, ExperimentTracker, DataStore, LLMClient
from rdkit_core.config import load_settings
```
