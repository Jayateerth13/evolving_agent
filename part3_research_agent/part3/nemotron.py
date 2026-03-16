"""Nemotron (NVIDIA NIM) client for Part 3: plan search and rank papers."""

from __future__ import annotations

import json
import os
import httpx

NIM_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _chat(messages: list[dict], api_key: str) -> str:
    """Call Nemotron chat completions. Returns assistant content."""
    r = httpx.post(
        f"{NIM_BASE}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
        },
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return content.strip()


def nemotron_plan(user_query: str, api_key: str) -> tuple[str, list[str]]:
    """Turn user input into a search query and preferred paper sources.
    Returns (search_query, preferred_sources)."""
    system = """You are a research paper scout. Given the user's project or topic, output a short search query and which sources to use.
Respond with ONLY a valid JSON object, no other text. Use this exact shape:
{"search_query": "short search phrase", "preferred_sources": ["semantic_scholar", "arxiv", "openalex"]}
Allowed sources: semantic_scholar, arxiv, openalex. Include at least one."""
    content = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ],
        api_key,
    )
    try:
        parsed = json.loads(content)
        q = parsed.get("search_query") or user_query
        src = parsed.get("preferred_sources")
        if isinstance(src, list):
            return (q, [s for s in src if s in ("semantic_scholar", "arxiv", "openalex")])
        return (q, ["semantic_scholar", "arxiv", "openalex"])
    except (json.JSONDecodeError, TypeError):
        return (user_query, ["semantic_scholar", "arxiv"])


def nemotron_rank(
    user_query: str,
    candidates: list[dict],
    api_key: str,
    top_k: int = 5,
) -> tuple[list[int], str]:
    """Rank paper candidates and pick top_k. Returns (selected_indices, summary)."""
    if not candidates:
        return ([], "No papers found.")

    lines = []
    for i, c in enumerate(candidates):
        title = c.get("title", "")
        source = c.get("source", "")
        paper_id = c.get("paper_id", "")
        abstract = (c.get("abstract") or "")[:200]
        lines.append(f"[{i}] source={source} paper_id={paper_id} title={title} abstract={abstract}")

    system = f"""You are a research paper scout. Rank these papers by relevance to the user's topic. Pick the top {top_k}.
Respond with ONLY a valid JSON object, no other text:
{{"selected_indices": [0, 1, 2, ...], "summary": "One sentence explaining why these papers are relevant."}}
Use the [index] numbers from the list. selected_indices must be an array of up to {top_k} integers."""

    user_msg = f"User topic: {user_query}\n\nPapers:\n" + "\n".join(lines)
    content = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        api_key,
    )

    try:
        parsed = json.loads(content)
        indices = parsed.get("selected_indices")
        if isinstance(indices, list):
            indices = [int(x) for x in indices if isinstance(x, (int, float)) and 0 <= int(x) < len(candidates)][:top_k]
        else:
            indices = list(range(min(top_k, len(candidates))))
        summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else "Selected papers relevant to your topic."
        return (indices, summary)
    except (json.JSONDecodeError, TypeError, ValueError):
        return (list(range(min(top_k, len(candidates)))), "Selected papers relevant to your topic.")
