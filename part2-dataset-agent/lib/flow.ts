/**
 * Part 2 Dataset Agent: full flow from user input to Part2Output.
 */

import type { DatasetCandidate } from "./sources/kaggle";
import type { Part2Dataset, Part2Output } from "./types";
import { nemotronPlan, nemotronRank } from "./nemotron";
import { searchKaggle } from "./sources/kaggle";
import { searchGoogleDatasetSearch } from "./sources/googleDatasetSearch";

export interface FlowEnv {
  NIM_API_KEY: string;
  KAGGLE_USERNAME?: string;
  KAGGLE_KEY?: string;
  GOOGLE_API_KEY?: string;
  GOOGLE_CSE_ID?: string;
}

export async function runDatasetAgent(
  userInput: string,
  env: FlowEnv
): Promise<Part2Output> {
  const search_queries_used: string[] = [];

  // Step 1: Nemotron plan
  const plan = await nemotronPlan(userInput, env.NIM_API_KEY);
  const query = plan.search_query.trim() || userInput;
  search_queries_used.push(query);

  // Step 2: Call sources
  const sources = plan.preferred_sources.map((s) => s.toLowerCase());
  const allCandidates: DatasetCandidate[] = [];

  if (sources.includes("kaggle") && env.KAGGLE_USERNAME && env.KAGGLE_KEY) {
    try {
      const kaggle = await searchKaggle(query, {
        username: env.KAGGLE_USERNAME,
        key: env.KAGGLE_KEY,
        maxResults: 10,
      });
      allCandidates.push(...kaggle);
    } catch (e) {
      console.error("Kaggle search failed:", e);
    }
  }

  if (sources.includes("google_dataset_search")) {
    const google = await searchGoogleDatasetSearch(query, {
      apiKey: env.GOOGLE_API_KEY,
      cseId: env.GOOGLE_CSE_ID,
      maxResults: 5,
    });
    allCandidates.push(...google);
  }

  if (allCandidates.length === 0) {
    return {
      datasets: [],
      search_queries_used,
      summary: "No datasets found. Add Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY) or Google Dataset Search (GOOGLE_API_KEY, GOOGLE_CSE_ID).",
    };
  }

  // Step 3: Nemotron rank
  const { selected_slugs, summary } = await nemotronRank(
    userInput,
    allCandidates,
    env.NIM_API_KEY
  );

  // Step 4 & 5: Build output (we return metadata + URLs; actual download can be done by Part 4 or frontend)
  const byKey = (c: DatasetCandidate) => `${c.source}:${c.slug_or_id}`;
  const selectedSet = new Set(selected_slugs.map((s) => `${s.source}:${s.slug_or_id}`));
  const selected = allCandidates.filter((c) => selectedSet.has(byKey(c)));

  const datasets: Part2Dataset[] = selected.map((c) => ({
    source: c.source,
    name: c.name,
    slug_or_id: c.slug_or_id,
    path_or_url: c.url || c.slug_or_id,
    format: (c.format as Part2Dataset["format"]) || "unknown",
    description: c.description,
    size: c.size,
  }));

  return {
    datasets,
    search_queries_used,
    summary,
  };
}
