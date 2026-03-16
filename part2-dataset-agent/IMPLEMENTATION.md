# Part 2: Dataset Agent – What Was Implemented

This folder contains **Part 2 (Dataset Agent)** of the multi-agent hackathon project. It takes **user input** (e.g. “I want to build house price prediction”), uses **NVIDIA Nemotron** to plan and rank, and pulls **dataset suggestions** from **Kaggle** and **Google Dataset Search**.

---

## What You Have

### 1. **API**
- **`POST /api/datasets`**  
  - **Body:** `{ "user_input": "your project idea" }`  
  - **Response:** Part 2 output (see below) for integration with Part 3, Part 4, and the single app.

### 2. **Flow (5 steps)**
1. **Nemotron plan** – Turns user input into a search query and preferred sources (`kaggle`, `google_dataset_search`).
2. **Source calls** – Your backend calls Kaggle API and/or Google Custom Search with that query.
3. **Nemotron rank** – Nemotron ranks the raw results and picks the top 1–2 datasets.
4. **Output shape** – Builds the response so the rest of the app can consume it.

### 3. **Response contract (for integration)**
```ts
{
  "datasets": [
    {
      "source": "kaggle",
      "name": "Dataset title",
      "slug_or_id": "owner/dataset-slug",
      "path_or_url": "https://www.kaggle.com/datasets/...",
      "format": "csv",
      "description": "...",
      "size": "..."
    }
  ],
  "search_queries_used": ["house price prediction"],
  "summary": "One sentence from Nemotron explaining why these datasets fit."
}
```

### 4. **Files**
| Path | Purpose |
|------|--------|
| `app/api/datasets/route.ts` | POST handler; reads `user_input`, calls flow, returns Part 2 output. |
| `app/page.tsx` | Simple UI: textarea + “Find datasets” → shows JSON result. |
| `lib/flow.ts` | Orchestrates plan → search → rank → output. |
| `lib/nemotron.ts` | Nemotron (NIM) client: `nemotronPlan()`, `nemotronRank()`. |
| `lib/types.ts` | `Part2Output`, `Part2Dataset`, `NemotronPlan`, etc. |
| `lib/sources/kaggle.ts` | Kaggle REST: list datasets by search query. |
| `lib/sources/googleDatasetSearch.ts` | Google Custom Search for Dataset Search (optional). |
| `.env.example` | Env vars you need (see below). |

### 5. **Nemotron**
- **Model:** `nvidia/nemotron-3-super-120b-a12b`
- **Endpoint:** `https://integrate.api.nvidia.com/v1/chat/completions`
- **Auth:** `Authorization: Bearer <NVIDIA_API_KEY>`
- **Usage:** Step 1 (plan) and Step 3 (rank). No tool calling in this implementation; the backend calls Kaggle/Google and then sends results to Nemotron for ranking.

---

## What You Need to Do

### Required
1. **NVIDIA API key (Nemotron)**  
   - Get it at [build.nvidia.com](https://build.nvidia.com) for the Nemotron model (e.g. “view code” for `nemotron-3-super-120b-a12b`).  
   - In `.env.local`:  
     `NVIDIA_API_KEY=your_key`  
     (or `NIM_API_KEY=your_key`; the route checks both.)

2. **Kaggle credentials**  
   - [Kaggle Account](https://www.kaggle.com) → Settings → “Create API Token” (downloads `kaggle.json`).  
   - In `.env.local`:  
     `KAGGLE_USERNAME=<username>`  
     `KAGGLE_KEY=<key>`

### Optional
3. **Google Dataset Search**  
   - Create a [Custom Search Engine](https://programmablesearchengine.google.com/) that includes `datasetsearch.research.google.com`.  
   - Create an [API key](https://console.cloud.google.com/apis/credentials) for Custom Search API.  
   - In `.env.local`:  
     `GOOGLE_API_KEY=...`  
     `GOOGLE_CSE_ID=...`  
   - If these are missing, Part 2 still runs using only Kaggle.

---

## Run and Deploy

```bash
cd part2-dataset-agent
cp .env.example .env.local
# Edit .env.local with your keys

npm install
npm run dev
```

- **Local:** Open [http://localhost:3000](http://localhost:3000), type a project idea, click “Find datasets”.  
- **API only:** `curl -X POST http://localhost:3000/api/datasets -H "Content-Type: application/json" -d '{"user_input":"house price prediction"}'`

**Vercel:** Push this folder (or the repo) and connect to Vercel. Add the same env vars in the Vercel project settings. The app is a standard Next.js app and deploys as-is.

---

## Integration With Part 3 and Part 4

- Part 2 returns the JSON above.  
- Your single app (or orchestrator) should:
  - Call `POST /api/datasets` with `user_input`.
  - Pass `result.datasets`, `result.summary`, and `result.search_queries_used` to Part 3 (research) and Part 4 (coding) as needed.  
- No shared state beyond this response; the “single app” can be a Next.js app that hosts Part 2, Part 3, and Part 4 routes and a single frontend that chains them.

---

## Summary

- **Part 2 is implemented** as a Next.js app with one API route and a minimal UI.  
- **Nemotron** is used for planning (query + sources) and ranking (top datasets).  
- **Kaggle** is the main data source; **Google Dataset Search** is optional.  
- **You need to provide:** NVIDIA API key and Kaggle username/key in `.env.local` (and optionally Google keys).  
- **Output** is a fixed contract so the rest of the team can integrate Part 3 and Part 4 and deploy everything as a single app on Vercel.
