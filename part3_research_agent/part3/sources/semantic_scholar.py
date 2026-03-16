"""Semantic Scholar API: search papers by query."""

from __future__ import annotations

import httpx
from part3.types import PaperResult

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,authors,year,url,citationCount,openAccessPdf"


def search_semantic_scholar(query: str, *, limit: int = 10) -> list[PaperResult]:
    """Search Semantic Scholar for papers. Returns list of PaperResult dicts."""
    url = f"{BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": FIELDS}
    try:
        r = httpx.get(url, params=params, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    papers: list[PaperResult] = []
    for hit in data.get("data", [])[:limit]:
        pid = hit.get("paperId") or ""
        title = hit.get("title") or ""
        abstract = (hit.get("abstract") or "")[:500]
        authors = ", ".join(a.get("name", "") for a in hit.get("authors", []))[:200]
        year = str(hit.get("year", "")) if hit.get("year") else ""
        url_str = hit.get("url") or f"https://www.semanticscholar.org/paper/{pid}"
        pdf = ""
        if hit.get("openAccessPdf"):
            pdf = hit["openAccessPdf"].get("url", "") or ""
        papers.append(
            PaperResult(
                source="semantic_scholar",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url_str,
                pdf_url=pdf,
                year=year,
                citation_count=hit.get("citationCount") or 0,
                paper_id=pid,
            )
        )
    return papers
