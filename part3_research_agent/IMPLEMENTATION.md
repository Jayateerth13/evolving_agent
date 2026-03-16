# Part 3: Research Paper Agent – What Was Implemented

Part 3 takes an **input query** (topic or project idea), uses **Nemotron** to plan and rank, and fetches **relevant research papers** from **Semantic Scholar** and **arXiv**. Results are returned in a fixed contract and can be **displayed** in the included UI or consumed by the single app.

---

## What You Have

### 1. **API**
- **`POST /api/papers`**  
  - **Body:** `{ "query": "your topic" }` or `{ "user_input": "..." }`  
  - **Response:** Part 3 output (papers list + summary + queries used).

- **`GET /`** – Web form; **`POST /`** – same form submits query and shows results on the page.

### 2. **Flow**
1. **Nemotron plan** – Converts user query into a search query and preferred sources (`semantic_scholar`, `arxiv`).
2. **Source calls** – Fetches from Semantic Scholar API and arXiv API.
3. **Nemotron rank** – Ranks candidates and selects top 5 (configurable).
4. **Output** – Builds the response in the shared contract.

### 3. **Response contract**
```json
{
  "papers": [
    {
      "source": "semantic_scholar",
      "title": "...",
      "authors": "...",
      "abstract": "...",
      "url": "...",
      "pdf_url": "...",
      "year": "...",
      "citation_count": 0,
      "paper_id": "..."
    }
  ],
  "search_queries_used": ["..."],
  "summary": "One sentence from Nemotron."
}
```

### 4. **Files**
| Path | Purpose |
|------|--------|
| `main.py` | FastAPI app: `/`, `POST /`, `POST /api/papers`. |
| `templates/index.html` | Search form and display of papers. |
| `part3/flow.py` | Orchestrates plan → search → rank → output. |
| `part3/nemotron.py` | Nemotron (NIM): `nemotron_plan()`, `nemotron_rank()`. |
| `part3/types.py` | `Part3Output`, `PaperResult`. |
| `part3/sources/semantic_scholar.py` | Semantic Scholar API search. |
| `part3/sources/arxiv.py` | arXiv API search. |
| `part3/sources/openalex.py` | OpenAlex API search. |
| `.env.example` | Env vars (NVIDIA_API_KEY or NIM_API_KEY). |

### 5. **Paper sources**
- **Semantic Scholar** – `https://api.semanticscholar.org/graph/v1/paper/search` (no key required for basic use).
- **arXiv** – `http://export.arxiv.org/api/query` (no key required).
- **OpenAlex** – `https://api.openalex.org/works` (no key required for basic search).

### 6. **Nemotron**
- **Model:** `nvidia/nemotron-3-super-120b-a12b`
- **Endpoint:** `https://integrate.api.nvidia.com/v1/chat/completions`
- **Auth:** `Authorization: Bearer <NVIDIA_API_KEY>` (or `NIM_API_KEY`)

---

## What You Need

- **NVIDIA API key** for Nemotron (from [build.nvidia.com](https://build.nvidia.com)). Set `NVIDIA_API_KEY` or `NIM_API_KEY` in `.env` (or environment).

No API keys are required for Semantic Scholar or arXiv for normal usage.

---

## Run and display

```bash
cd part3_research_agent
pip install -r requirements.txt
cp .env.example .env   # add NVIDIA_API_KEY
uvicorn main:app --reload
```

- **Browser:** http://127.0.0.1:8000 → enter query → “Find research papers” → results shown on the page.
- **API:** `curl -X POST http://127.0.0.1:8000/api/papers -H "Content-Type: application/json" -d '{"query":"your topic"}'`

---

## Integration with Part 2 and Part 4

- Part 3 returns the JSON above. The single app can call `POST /api/papers` with the same (or derived) query used for Part 2, then pass `papers` and `summary` to Part 4 (coding agent).
- Each paper has `url`, `pdf_url`, `title`, `abstract`, `source` for display or downstream use.
