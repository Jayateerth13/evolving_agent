"""OpenAlex API: search works (papers) by query."""

from __future__ import annotations

import httpx
from part3.types import PaperResult

BASE = "https://api.openalex.org/works"


def search_openalex(query: str, *, per_page: int = 10) -> list[PaperResult]:
    """Search OpenAlex for papers. Returns list of PaperResult dicts."""
    params = {"search": query, "per_page": per_page}
    try:
        r = httpx.get(BASE, params=params, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    papers: list[PaperResult] = []
    for hit in data.get("results", [])[:per_page]:
        title = (hit.get("title") or "").strip()
        abstract = ""
        if not hit.get("abstract_inverted_index"):
            abstract = ""
        else:
            abstract = "(See link for full abstract)"
        authors_list = hit.get("authorships") or []
        authors = ", ".join(
            (a.get("author", {}).get("display_name") or "")
            for a in authors_list[:10]
        )[:200]
        year = ""
        if hit.get("publication_year"):
            year = str(hit["publication_year"])
        url_str = hit.get("doi") and f"https://doi.org/{hit['doi']}" or hit.get("id", "")
        pdf_url = ""
        for loc in hit.get("locations", []) or []:
            if loc.get("is_oa") and loc.get("pdf_url"):
                pdf_url = loc.get("pdf_url", "") or ""
                break
        cited_by = hit.get("cited_by_count") or 0
        oa_id = (hit.get("id") or "").replace("https://openalex.org/", "")
        papers.append(
            PaperResult(
                source="openalex",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url_str,
                pdf_url=pdf_url,
                year=year,
                citation_count=cited_by,
                paper_id=oa_id,
            )
        )
    return papers
