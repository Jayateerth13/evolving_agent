/**
 * Nemotron (NVIDIA NIM) client for Part 2: plan search and rank datasets.
 * Model: nvidia/nemotron-3-super-120b-a12b @ https://integrate.api.nvidia.com/v1
 */

const NIM_BASE = "https://integrate.api.nvidia.com/v1";
const MODEL = "nvidia/nemotron-3-super-120b-a12b";

export interface NemotronMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

async function chat(messages: NemotronMessage[], apiKey: string): Promise<string> {
  const res = await fetch(`${NIM_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      max_tokens: 2048,
      temperature: 0.3,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Nemotron API error ${res.status}: ${err}`);
  }

  const data = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = data.choices?.[0]?.message?.content ?? "";
  return content.trim();
}

/**
 * Step 1: Ask Nemotron to turn user input into a search plan (query + preferred sources).
 */
export async function nemotronPlan(
  userInput: string,
  apiKey: string
): Promise<{ search_query: string; preferred_sources: string[] }> {
  const system = `You are a dataset scout agent. Given the user's project idea, output a short search query and which sources to use.
Respond with ONLY a valid JSON object, no other text. Use this exact shape:
{"search_query": "short search phrase", "preferred_sources": ["kaggle", "google_dataset_search"]}
Allowed sources: kaggle, google_dataset_search. Include at least one.`;

  const content = await chat(
    [
      { role: "system", content: system },
      { role: "user", content: userInput },
    ],
    apiKey
  );

  try {
    const parsed = JSON.parse(content) as { search_query?: string; preferred_sources?: string[] };
    return {
      search_query: typeof parsed.search_query === "string" ? parsed.search_query : userInput,
      preferred_sources: Array.isArray(parsed.preferred_sources)
        ? parsed.preferred_sources
        : ["kaggle"],
    };
  } catch {
    return { search_query: userInput, preferred_sources: ["kaggle"] };
  }
}

/**
 * Step 3: Ask Nemotron to rank candidates and pick top 1–2 for the user's project.
 */
export async function nemotronRank(
  userInput: string,
  candidates: Array<{ source: string; name: string; slug_or_id: string; description?: string }>,
  apiKey: string
): Promise<{ selected_slugs: Array<{ source: string; slug_or_id: string }>; summary: string }> {
  if (candidates.length === 0) {
    return { selected_slugs: [], summary: "No datasets found." };
  }

  const list = candidates
    .map(
      (c, i) =>
        `[${i}] source=${c.source} slug_or_id=${c.slug_or_id} name=${c.name} ${c.description ? `description=${c.description.slice(0, 200)}` : ""}`
    )
    .join("\n");

  const system = `You are a dataset scout. Rank these datasets by relevance to the user's project. Pick the top 1 or 2.
Respond with ONLY a valid JSON object, no other text:
{"selected_indices": [0, 1], "summary": "One sentence explaining why these datasets fit the project."}
Use the [index] numbers from the list. selected_indices must be an array of integers.`;

  const userMsg = `User project: ${userInput}\n\nDatasets:\n${list}`;
  const content = await chat(
    [
      { role: "system", content: system },
      { role: "user", content: userMsg },
    ],
    apiKey
  );

  try {
    const parsed = JSON.parse(content) as {
      selected_indices?: number[];
      summary?: string;
    };
    const indices = Array.isArray(parsed.selected_indices) ? parsed.selected_indices : [0];
    const selected_slugs = indices
      .filter((i) => i >= 0 && i < candidates.length)
      .map((i) => ({
        source: candidates[i].source,
        slug_or_id: candidates[i].slug_or_id,
      }));
    const summary =
      typeof parsed.summary === "string" ? parsed.summary : "Selected datasets for your project.";
    return { selected_slugs, summary };
  } catch {
    return {
      selected_slugs: candidates.slice(0, 2).map((c) => ({ source: c.source, slug_or_id: c.slug_or_id })),
      summary: "Selected top datasets for your project.",
    };
  }
}
