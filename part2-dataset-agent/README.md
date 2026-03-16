# Part 2: Dataset Agent

Agent that takes **user input** and pulls **dataset suggestions** from Kaggle and Google Dataset Search using **NVIDIA Nemotron** to plan and rank.

- **API:** `POST /api/datasets` with `{ "user_input": "your project idea" }`
- **Details:** See [IMPLEMENTATION.md](./IMPLEMENTATION.md) for what was built and what you need to provide (API keys, etc.).

```bash
cp .env.example .env.local   # add your keys
npm install
npm run dev
```

Then open http://localhost:3000 or call the API directly.
