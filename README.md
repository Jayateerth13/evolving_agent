# R&D Multi-Agent ML System

A production multi-agent system that autonomously iterates toward the best possible model for tabular/structured data. Three specialised agents — **Data Mining**, **Research Copilot**, and **Kaggle** — are coordinated by an **Orchestrator** that routes between them based on the current experiment state.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Orchestrator                         │
│              (polls experiment_context.json)               │
│                                                          │
│   status=running    status=stalled    status=staged      │
│       │                  │                 │              │
│       ▼                  ▼                 ▼              │
│  ┌──────────┐   ┌───────────────┐   ┌────────────┐      │
│  │  Data     │   │   Research    │   │   Kaggle   │      │
│  │  Mining   │   │   Copilot    │   │   Agent    │      │
│  │  Agent    │   │   Agent      │   │            │      │
│  └──────────┘   └───────────────┘   └────────────┘      │
│       │                  │                 │              │
│       └──────────────────┴─────────────────┘              │
│                          │                                │
│              experiment_context.json                       │
│              (shared boundary contract)                    │
└──────────────────────────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  rdkit-core  │
                    │  (Part 1)    │
                    └─────────────┘
```

## Repository Structure

```
evolving_agent/
├── part1_core/                 # rdkit-core pip package (foundation)
│   ├── src/rdkit_core/
│   │   ├── models/             # Pydantic contracts
│   │   ├── tools/              # Executor, Tracker, DataStore, Registry, LLM
│   │   └── schemas/            # Exported JSON schemas
│   ├── tests/
│   └── pyproject.toml
├── part2_data_mining/          # Data Mining Agent
│   └── agent/
├── part3_research_copilot/     # Research Copilot Agent
│   └── agent/
├── part4_kaggle_orchestrator/  # Kaggle Agent + Orchestrator
│   ├── kaggle_agent/
│   └── orchestrator/
├── experiments/generated/      # Model modules from Research Copilot
├── data/versions/              # Versioned datasets
├── experiment_context.json     # Shared boundary contract
└── config.yaml                 # Global configuration
```

## Quick Start

```bash
# 1. Install Part 1 (foundation)
cd part1_core
pip install -e ".[dev]"

# 2. Export JSON Schema
python scripts/export_schema.py

# 3. Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# 4. Set your NVIDIA API key
export NVIDIA_API_KEY="nvapi-..."

# 5. Run tests
pytest
```

## Communication Contract

All agents communicate exclusively via `experiment_context.json`. The Orchestrator dispatches agents as **subprocesses** based on the `status` field:

| Status | Agent Dispatched | Next Status |
|--------|-----------------|-------------|
| `running` | Data Mining Agent | `stalled` or `staged` |
| `stalled` | Research Copilot | `running` |
| `staged` | Kaggle Agent | `promoted` or `rejected` |
| `promoted` | — (pipeline complete) | — |
| `rejected` | — (pipeline complete) | — |

## Tech Stack

- **LLM**: NVIDIA Nemotron (via OpenAI-compatible API)
- **Tracking**: MLflow
- **Execution**: Docker sandbox / local subprocess
- **Data**: Pandas + Parquet + content-hashed versioning
- **Config**: Pydantic v2 + YAML
