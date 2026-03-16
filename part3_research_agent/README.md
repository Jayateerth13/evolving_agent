# Part 3: Research Paper Agent

Python agent that takes an **input query** and fetches **relevant research papers** from **Semantic Scholar**, **arXiv**, and **OpenAlex**, using **NVIDIA Nemotron** to plan the search and rank results.

- **Web UI:** Open `/`, enter a topic, click "Find research papers" to see results.
- **API:** `POST /api/papers` with `{"query": "your topic"}` or `{"user_input": "..."}`.

## Setup

```bash
cd part3_research_agent
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set NVIDIA_API_KEY (or NIM_API_KEY)
```

## Run

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 and use the form, or:

```bash
curl -X POST http://127.0.0.1:8000/api/papers -H "Content-Type: application/json" -d '{"query":"transformer tabular data"}'
```

See [IMPLEMENTATION.md](./IMPLEMENTATION.md) for details and integration.
