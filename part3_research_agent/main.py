"""Part 3 Research Agent: FastAPI app and display."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from part3.flow import run_research_agent
from part3.types import Part3Output

app = FastAPI(title="Part 3: Research Paper Agent")
DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(DIR / "templates"))


def _api_key() -> str:
    return os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY") or ""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the search form and results display."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def search_post(
    request: Request,
    query: str = Form(default=""),
):
    """Form POST: run research agent and render results in the same page."""
    query = (query or "").strip()
    api_key = _api_key()
    if not api_key:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Missing NVIDIA_API_KEY or NIM_API_KEY.", "result": None},
        )
    if not query:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Enter a search query.", "result": None},
        )
    try:
        result = run_research_agent(query, api_key=api_key)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": None, "result": result, "query": query},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "result": None, "query": query},
        )


@app.post("/api/papers")
async def api_papers(request: Request) -> JSONResponse:
    """JSON API: POST body { \"query\": \"...\" } -> Part3Output."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
    query = (body.get("query") or body.get("user_input") or "").strip()
    api_key = _api_key()
    if not api_key:
        return JSONResponse(
            {"error": "Missing NVIDIA_API_KEY or NIM_API_KEY."},
            status_code=503,
        )
    if not query:
        return JSONResponse(
            {"error": "Missing 'query' or 'user_input' in body."},
            status_code=400,
        )
    try:
        result: Part3Output = run_research_agent(query, api_key=api_key)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )
