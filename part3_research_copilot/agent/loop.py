"""Main Research Copilot loop.

Reads experiment_context.json (status must be 'stalled'), searches for
relevant papers based on failed hypotheses, extracts a model spec,
generates a runnable Python module, validates it, and resets the
context back to 'running' so the Data Mining Agent can resume.
"""

from __future__ import annotations

from pathlib import Path

from rdkit_core import ExperimentContext, Status
from rdkit_core.config import load_settings
from rdkit_core.tools.executor import LocalExecutor
from rdkit_core.tools.llm_client import LLMClient

from .codegen import generate_model_code, save_model_module, validate_code
from .paper_search import search_papers
from .spec_extractor import extract_model_spec

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _header(text: str) -> None:
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}{RESET}")


def _step(icon: str, text: str) -> None:
    print(f"  {icon} {text}")


def run_research_copilot(
    context_path: str = "experiment_context.json",
    config_path: str = "config.yaml",
    max_retries: int = 3,
) -> ExperimentContext:
    """Run the Research Copilot agent."""

    settings = load_settings(config_path)

    _header("Research Copilot — Activating")

    # ── Step 1: Read context + verify stalled ────────────────────
    ctx = ExperimentContext.load(context_path)

    if ctx.status != Status.STALLED:
        _step("⏭️", f"Status is '{ctx.status.value}', not 'stalled' — nothing to do")
        return ctx

    _step("📖", f"Context loaded: {ctx.iteration} iterations, {len(ctx.hypothesis_log)} hypotheses")
    _step("🛑", f"Pipeline stalled at stall_count={ctx.stall_count}")

    # ── Step 2: Build search query from failed hypotheses ────────
    _header("Step 1 — Deriving Search Query")

    recent = ctx.hypothesis_log[-3:] if ctx.hypothesis_log else []
    tried = [h.hypothesis for h in recent]

    llm = LLMClient(
        model=settings.llm.model,
        provider=settings.llm.provider,
        temperature=settings.llm.temperature,
    )

    search_query = _build_search_query(tried, llm)
    _step("🔍", f"Search query: \"{search_query}\"")

    # ── Step 3: Search papers ────────────────────────────────────
    _header("Step 2 — Searching Papers")

    papers = search_papers(search_query, max_per_source=5)
    _step("📚", f"Found {len(papers)} papers")

    if not papers:
        _step("⚠️ ", "No papers found — using fallback approach")

    for i, p in enumerate(papers[:3]):
        cite = f"{p.citation_count} cites" if p.citation_count else ""
        year = f"({p.year})" if p.year else ""
        _step(f"  {i+1}.", f"{p.title[:70]} {DIM}{year} {cite}{RESET}")

    top_paper = papers[0] if papers else None

    # ── Step 4: Extract model spec ───────────────────────────────
    _header("Step 3 — Extracting Model Spec")

    dataset_context = (
        f"Tabular {settings.dataset.task_type} task, "
        f"target='{settings.dataset.target_column}', "
        f"metric={settings.dataset.primary_metric}"
    )

    if top_paper:
        _step("📄", f"Analyzing: {top_paper.title[:60]}")
        spec = extract_model_spec(top_paper, dataset_context, llm)
    else:
        from rdkit_core.models.spec import PaperResult as PR
        dummy = PR(title="Enhanced Gradient Boosting for Tabular Data")
        spec = extract_model_spec(dummy, dataset_context, llm)

    _step("🏗️", f"Architecture: {spec.architecture}")
    _step("⚙️", f"Framework: {spec.framework}")
    if spec.hyperparameters:
        _step("🔧", f"Hyperparams: {spec.hyperparameters}")
    if spec.notes:
        _step("💡", f"Notes: {spec.notes}")

    # ── Step 5: Generate + validate code ─────────────────────────
    _header("Step 4 — Generating Model Code")

    executor = LocalExecutor()
    schema_str = f"Target: {settings.dataset.target_column}, Task: {settings.dataset.task_type}"
    code = ""
    valid = False

    for attempt in range(max_retries):
        _step("🖥️", f"Generating code (attempt {attempt + 1}/{max_retries})...")
        code = generate_model_code(spec, schema_str, llm)

        _step("🧪", "Validating in sandbox...")
        valid, err = validate_code(code, executor)

        if valid:
            _step(GREEN + "✅", f"Validation passed!{RESET}")
            break
        else:
            _step(YELLOW + "❌", f"Validation failed: {err[:100]}{RESET}")

    if not valid:
        _step("⚠️ ", "All retries failed — using fallback code")
        from .codegen import _fallback_code
        code = _fallback_code(spec)

    # ── Step 6: Save module ──────────────────────────────────────
    path = save_model_module(code, spec)
    _step("💾", f"Model saved: {path}")

    # ── Step 7: Update context + reset ───────────────────────────
    _header("Step 5 — Resetting Pipeline")

    if top_paper:
        ref = top_paper.arxiv_id or top_paper.doi or top_paper.title[:50]
        ctx.paper_refs.append(ref)
        _step("📎", f"Citation: {ref}")

    ctx.model_spec = spec.model_dump()
    ctx.reset_for_new_iteration()
    ctx.save(context_path)

    _step("🔄", f"Status reset to '{ctx.status.value}', stall_count={ctx.stall_count}")
    _step("➡️ ", "Data Mining Agent will resume with the new model")

    return ctx


def _build_search_query(tried_hypotheses: list[str], llm: LLMClient) -> str:
    """Derive a paper search query from failed hypotheses via Nemotron."""
    base = "tabular data machine learning"

    try:
        if not llm._client.api_key:
            raise ValueError("No API key")
    except Exception:
        return f"{base} advanced feature engineering gradient boosting"

    system = """Given these failed ML experiment hypotheses on a tabular dataset,
suggest a short academic search query (5-10 words) to find papers with
new techniques that might help. Return ONLY the search query string, nothing else."""

    tried_text = "\n".join(f"- {h}" for h in tried_hypotheses) or "- basic feature engineering"

    try:
        query = llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Failed hypotheses:\n{tried_text}"},
            ],
            temperature=0.4,
        )
        return query.strip().strip('"').strip("'")[:100]
    except Exception:
        return f"{base} feature engineering gradient boosting"
