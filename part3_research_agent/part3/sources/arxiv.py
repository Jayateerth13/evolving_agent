"""arXiv API: search papers by query."""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
import httpx
from part3.types import PaperResult

NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, *, max_results: int = 10) -> list[PaperResult]:
    """Search arXiv for papers. Returns list of PaperResult dicts."""
    q = urllib.parse.quote(f"all:{query}")
    url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={max_results}"
    try:
        r = httpx.get(url, timeout=15.0)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception:
        return []

    papers: list[PaperResult] = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        title = (title_el.text or "").strip().replace("\n", " ")
        summary_el = entry.find("atom:summary", NS)
        abstract = (summary_el.text or "").strip().replace("\n", " ")[:500]
        link_el = entry.find("atom:id", NS)
        url_str = (link_el.text or "").strip()
        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "") or ""
                break
        authors = ", ".join(
            a.find("atom:name", NS).text or ""
            for a in entry.findall("atom:author", NS)
        )[:200]
        published = entry.find("atom:published", NS)
        year = (published.text or "")[:4] if published is not None else ""
        arxiv_id = url_str.split("/")[-1] if url_str else ""
        papers.append(
            PaperResult(
                source="arxiv",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url_str,
                pdf_url=pdf_url,
                year=year,
                citation_count=0,
                paper_id=arxiv_id,
            )
        )
    return papers
