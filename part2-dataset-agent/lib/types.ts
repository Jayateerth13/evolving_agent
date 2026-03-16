/**
 * Part 2 (Dataset Agent) output contract for integration with Part 3, Part 4, and the single app.
 */
export interface Part2Dataset {
  source: "kaggle" | "google_dataset_search" | "huggingface" | string;
  name: string;
  slug_or_id: string;
  path_or_url: string;
  format: "csv" | "json" | "parquet" | "unknown";
  description?: string;
  size?: string;
}

export interface Part2Output {
  datasets: Part2Dataset[];
  search_queries_used: string[];
  summary: string;
}

/** Raw result from a dataset source (Kaggle, Google, etc.) before ranking. */
export interface DatasetCandidate {
  source: string;
  name: string;
  slug_or_id: string;
  description?: string;
  url?: string;
  size?: string;
  format?: string;
}

/** Nemotron plan: what to search and where. */
export interface NemotronPlan {
  search_query: string;
  preferred_sources: string[];
  data_type?: string;
}
