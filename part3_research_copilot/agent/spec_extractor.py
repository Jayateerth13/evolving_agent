"""Extract a ModelSpec from a paper using Nemotron.

Uses the paper's title + abstract (no PDF parsing needed for demo).
Nemotron extracts architecture, hyperparameters, and preprocessing
steps into a structured ModelSpec.
"""

from __future__ import annotations

from rdkit_core.models.spec import ModelSpec, PaperResult
from rdkit_core.tools.llm_client import LLMClient


def extract_model_spec(
    paper: PaperResult,
    dataset_context: str,
    llm: LLMClient,
) -> ModelSpec:
    """Extract a ModelSpec from a paper's abstract via Nemotron.

    Falls back to a reasonable default if LLM is unavailable.
    """
    try:
        if not llm._client.api_key:
            raise ValueError("No API key")
    except Exception:
        return _fallback_spec(paper)

    system = """You are an ML research engineer. Given a paper's title and abstract,
extract a model specification for tabular/structured data.

Return ONLY valid JSON with these keys:
- architecture: short description of the model type (e.g. "gradient boosted trees with target encoding")
- framework: one of sklearn, pytorch, lightgbm, xgboost, catboost
- hyperparameters: dict of key hyperparameters mentioned or implied
- preprocessing_steps: list of preprocessing steps mentioned
- training_recipe: dict with training details (loss, optimizer, epochs, etc.)
- notes: one sentence on why this approach might help"""

    user_msg = (
        f"Paper: {paper.title}\n\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Dataset context: {dataset_context}\n\n"
        f"Extract a model spec that could be applied to this tabular dataset."
    )

    try:
        raw = llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return ModelSpec(
            paper_title=paper.title,
            paper_id=paper.arxiv_id or paper.doi or "",
            architecture=raw.get("architecture", ""),
            framework=raw.get("framework", "sklearn"),
            hyperparameters=raw.get("hyperparameters", {}),
            preprocessing_steps=raw.get("preprocessing_steps", []),
            training_recipe=raw.get("training_recipe", {}),
            notes=raw.get("notes", ""),
        )
    except Exception as e:
        print(f"  [warn] Spec extraction failed: {e}")
        return _fallback_spec(paper)


def _fallback_spec(paper: PaperResult) -> ModelSpec:
    """Deterministic fallback when LLM is unavailable."""
    return ModelSpec(
        paper_title=paper.title,
        paper_id=paper.arxiv_id or "",
        architecture="gradient_boosting_with_target_encoding",
        framework="lightgbm",
        hyperparameters={
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 8,
            "num_leaves": 63,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
        },
        preprocessing_steps=[
            "target_encode_high_cardinality",
            "fill_missing_with_median",
            "log_transform_skewed",
        ],
        training_recipe={"cv_folds": 5, "early_stopping_rounds": 50},
        notes="Fallback spec — enhanced LightGBM with target encoding and regularization",
    )
