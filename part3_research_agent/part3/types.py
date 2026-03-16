"""Part 3 output contract for integration with the single app."""

from typing import TypedDict


class PaperResult(TypedDict, total=False):
    """A single research paper for display and downstream use."""

    source: str
    title: str
    authors: str
    abstract: str
    url: str
    pdf_url: str
    year: str
    citation_count: int
    paper_id: str  # external ID (DOI, arXiv ID, etc.)


class Part3Output(TypedDict):
    """Response shape returned by Part 3 API."""

    papers: list[PaperResult]
    search_queries_used: list[str]
    summary: str
