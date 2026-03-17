"""Paper search across Semantic Scholar and ArXiv.

No API keys required for either service. Returns results as
rdkit-core PaperResult models.
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from rdkit_core.models.spec import PaperResult

_SS_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_FIELDS = "paperId,title,abstract,authors,year,url,citationCount,openAccessPdf"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_semantic_scholar(query: str, limit: int = 10) -> list[PaperResult]:
    try:
        r = httpx.get(
            f"{_SS_BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": _SS_FIELDS},
            timeout=15.0,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    results = []
    for hit in data.get("data", [])[:limit]:
        pid = hit.get("paperId", "")
        pdf_url = ""
        if hit.get("openAccessPdf"):
            pdf_url = hit["openAccessPdf"].get("url", "")
        results.append(PaperResult(
            title=hit.get("title", ""),
            authors=[a.get("name", "") for a in hit.get("authors", [])[:5]],
            abstract=(hit.get("abstract") or "")[:600],
            url=hit.get("url") or f"https://www.semanticscholar.org/paper/{pid}",
            pdf_url=pdf_url,
            year=hit.get("year") or 0,
            citation_count=hit.get("citationCount") or 0,
            has_code=False,
            source="semantic_scholar",
            arxiv_id=pid,
        ))
    return results


def search_arxiv(query: str, max_results: int = 10) -> list[PaperResult]:
    q = urllib.parse.quote(f"all:{query}")
    url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results={max_results}"
    try:
        r = httpx.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception:
        return []

    results = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title_el = entry.find("atom:title", _ARXIV_NS)
        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        summary_el = entry.find("atom:summary", _ARXIV_NS)
        abstract = (summary_el.text or "").strip().replace("\n", " ")[:600] if summary_el is not None else ""
        id_el = entry.find("atom:id", _ARXIV_NS)
        entry_url = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = entry_url.split("/")[-1] if entry_url else ""

        pdf_url = ""
        for link in entry.findall("atom:link", _ARXIV_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        authors = []
        for a in entry.findall("atom:author", _ARXIV_NS)[:5]:
            name_el = a.find("atom:name", _ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text)

        published = entry.find("atom:published", _ARXIV_NS)
        year = int((published.text or "0000")[:4]) if published is not None else 0

        results.append(PaperResult(
            title=title,
            authors=authors,
            abstract=abstract,
            url=entry_url,
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            year=year,
            citation_count=0,
            has_code=False,
            source="arxiv",
        ))
    return results


def search_papers(query: str, max_per_source: int = 5) -> list[PaperResult]:
    """Search both sources, merge and sort by citation count * recency."""
    papers = search_semantic_scholar(query, limit=max_per_source)
    papers += search_arxiv(query, max_results=max_per_source)

    def _score(p: PaperResult) -> float:
        recency = max(0, (p.year - 2018)) / 7.0 if p.year > 0 else 0.5
        citations = min(p.citation_count / 100.0, 1.0)
        return recency * 0.6 + citations * 0.4

    papers.sort(key=_score, reverse=True)
    return papers
