"""LLM-driven feature proposal and hypothesis generation via Nemotron.

All LLM calls go through rdkit-core's LLMClient. The proposer sends
the dataset profile and run history, and receives structured feature
engineering steps and next-iteration hypotheses.
"""

from __future__ import annotations

from typing import Any

from rdkit_core.models.spec import FeatureProposal, Hypothesis, ProfileReport, RunDiff
from rdkit_core.tools.llm_client import LLMClient


def _llm_available(llm: LLMClient) -> bool:
    """Check if the LLM client has a valid API key configured."""
    try:
        return bool(llm._client.api_key)
    except Exception:
        return False


def propose_features(
    profile: ProfileReport,
    schema_str: str,
    llm: LLMClient,
    strategy: str = "combined",
    top_k: int = 5,
) -> list[FeatureProposal]:
    """Ask Nemotron to propose feature engineering steps.

    Returns a ranked list of FeatureProposal with executable code snippets.
    Falls back to deterministic proposals if LLM is unavailable.
    """
    if not _llm_available(llm):
        print("  [info] No API key — using deterministic feature proposals")
        return _fallback_proposals(profile)

    system = f"""You are an expert ML feature engineer for tabular data.
Given the dataset profile and schema, propose {top_k} feature engineering steps.

Rules:
- Each step must be a single pandas operation on a DataFrame called `df`
- Use only pandas and numpy (imported as pd, np)
- Handle missing values, create interactions, encode categoricals
- Return ONLY valid JSON: a list of objects with keys: name, description, code_snippet, rationale, expected_uplift, priority, category
- code_snippet must be one executable line like: df['NewCol'] = df['A'] + df['B']
- category must be one of: missing_value, encoding, interaction, transform, aggregation
- priority: 1 = most important
- expected_uplift: estimated AUC/metric improvement (0.0 to 0.1)"""

    profile_text = (
        f"Dataset: {profile.n_rows} rows, {profile.n_cols} cols\n"
        f"Task: {profile.task_type}, Target: {profile.target_column}\n"
        f"Missing: {profile.missing_rates}\n"
        f"Cardinality: {profile.cardinality}\n"
        f"Types: {profile.dtypes}\n"
        f"Skew: {profile.skew_scores}\n"
        f"Correlation with target: {profile.correlation_with_target}\n"
    )

    user_msg = f"Dataset Profile:\n{profile_text}\n\nSchema:\n{schema_str}"

    try:
        raw = llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )

        items = raw if isinstance(raw, list) else raw.get("proposals", raw.get("features", [raw]))
        proposals = []
        for i, item in enumerate(items[:top_k]):
            proposals.append(FeatureProposal(
                name=item.get("name", f"feature_{i}"),
                description=item.get("description", ""),
                code_snippet=item.get("code_snippet", ""),
                rationale=item.get("rationale", ""),
                expected_uplift=float(item.get("expected_uplift", 0.0)),
                priority=int(item.get("priority", i + 1)),
                category=item.get("category", "transform"),
            ))
        return sorted(proposals, key=lambda p: p.priority)

    except Exception as e:
        print(f"  [warn] LLM feature proposal failed: {e}")
        return _fallback_proposals(profile)


def propose_next_step(
    run_diff: RunDiff | None,
    hypothesis_log: list[dict[str, Any]],
    profile: ProfileReport,
    llm: LLMClient,
) -> Hypothesis:
    """Ask Nemotron what to try next based on experiment history."""

    if not _llm_available(llm):
        print("  [info] No API key — using default hypothesis")
        return Hypothesis(
            description="Try additional feature interactions and hyperparameter tuning",
            rationale="Default strategy when LLM is unavailable",
            technique="feature_engineering",
        )

    system = """You are an ML experiment strategist for tabular data.
Given the run comparison and past hypotheses, propose ONE next experiment.

Return ONLY valid JSON with keys: description, rationale, estimated_uplift, priority, technique
- technique: one of feature_engineering, hyperparameter_tuning, model_change, ensemble, data_augmentation
- estimated_uplift: realistic estimate (0.001 to 0.05)"""

    history = "\n".join(
        f"  Iter {h.get('iteration', '?')}: {h.get('hypothesis', '?')} → {h.get('outcome', 'unknown')}"
        for h in hypothesis_log[-5:]
    ) or "  No previous experiments."

    diff_text = ""
    if run_diff:
        diff_text = f"Last run comparison: {run_diff.summary}\nMetric deltas: {run_diff.metric_deltas}"

    user_msg = (
        f"Dataset: {profile.task_type} task, {profile.n_rows} rows, {profile.n_cols} cols\n"
        f"Target: {profile.target_column}\n\n"
        f"Experiment history:\n{history}\n\n"
        f"{diff_text}"
    )

    try:
        raw = llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )
        return Hypothesis(
            description=raw.get("description", "Continue feature engineering"),
            rationale=raw.get("rationale", ""),
            estimated_uplift=float(raw.get("estimated_uplift", 0.01)),
            priority=int(raw.get("priority", 1)),
            technique=raw.get("technique", "feature_engineering"),
        )
    except Exception as e:
        print(f"  [warn] LLM hypothesis failed: {e}")
        return Hypothesis(
            description="Try hyperparameter tuning",
            rationale="Default fallback when LLM is unavailable",
            technique="hyperparameter_tuning",
        )


def _fallback_proposals(profile: ProfileReport) -> list[FeatureProposal]:
    """Deterministic fallback proposals when LLM is unavailable."""
    proposals = []

    for col, rate in profile.missing_rates.items():
        if rate > 0:
            fill = "median()" if profile.dtypes.get(col) == "numeric" else "mode()[0]"
            proposals.append(FeatureProposal(
                name=f"fill_{col}",
                description=f"Fill missing {col} with {fill}",
                code_snippet=f"df['{col}'] = df['{col}'].fillna(df['{col}'].{fill})",
                rationale=f"{col} has {rate*100:.1f}% missing values",
                expected_uplift=0.005,
                priority=len(proposals) + 1,
                category="missing_value",
            ))

    for col, dtype in profile.dtypes.items():
        if dtype == "categorical" and col != profile.target_column:
            proposals.append(FeatureProposal(
                name=f"encode_{col}",
                description=f"Label-encode {col}",
                code_snippet=f"df['{col}'] = df['{col}'].astype('category').cat.codes",
                rationale=f"Convert categorical {col} to numeric",
                expected_uplift=0.01,
                priority=len(proposals) + 1,
                category="encoding",
            ))

    return proposals[:5]
