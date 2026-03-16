"""Part 3 Research Agent: full flow from user query to Part3Output."""

from __future__ import annotations

from part3.types import PaperResult, Part3Output
from part3.nemotron import nemotron_plan, nemotron_rank
from part3.sources import search_semantic_scholar, search_arxiv, search_openalex


def run_research_agent(
    user_query: str,
    *,
    api_key: str,
    max_papers_per_source: int = 10,
    top_k: int = 5,
) -> Part3Output:
    """Run the research paper flow: plan -> search -> rank -> output."""
    search_queries_used: list[str] = []

    # Step 1: Nemotron plan
    search_query, preferred_sources = nemotron_plan(user_query, api_key)
    search_query = (search_query or user_query).strip()
    search_queries_used.append(search_query)

    # Step 2: Call sources
    all_candidates: list[PaperResult] = []
    sources = [s.lower() for s in preferred_sources]

    if "semantic_scholar" in sources:
        try:
            all_candidates.extend(
                search_semantic_scholar(search_query, limit=max_papers_per_source)
            )
        except Exception:
            pass

    if "arxiv" in sources:
        try:
            all_candidates.extend(
                search_arxiv(search_query, max_results=max_papers_per_source)
            )
        except Exception:
            pass

    if "openalex" in sources:
        try:
            all_candidates.extend(
                search_openalex(search_query, per_page=max_papers_per_source)
            )
        except Exception:
            pass

    if not all_candidates:
        return Part3Output(
            papers=[],
            search_queries_used=search_queries_used,
            summary="No papers found. Try a different query or check your connection.",
        )

    # Step 3: Nemotron rank
    candidates_as_dicts = [
        {
            "title": c.get("title"),
            "source": c.get("source"),
            "paper_id": c.get("paper_id"),
            "abstract": c.get("abstract"),
        }
        for c in all_candidates
    ]
    selected_indices, summary = nemotron_rank(
        user_query, candidates_as_dicts, api_key, top_k=top_k
    )

    # Step 4: Build output
    papers = [all_candidates[i] for i in selected_indices if 0 <= i < len(all_candidates)]

    return Part3Output(
        papers=papers,
        search_queries_used=search_queries_used,
        summary=summary,
    )
