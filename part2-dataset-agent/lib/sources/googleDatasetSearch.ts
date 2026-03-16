/**
 * Google Dataset Search via Custom Search JSON API.
 * Requires GOOGLE_API_KEY and GOOGLE_CSE_ID (create a custom search engine that includes datasetsearch.research.google.com).
 * If not configured, returns empty results so the flow still works with Kaggle only.
 */

const CSE_URL = "https://www.googleapis.com/customsearch/v1";

export interface DatasetCandidate {
  source: string;
  name: string;
  slug_or_id: string;
  description?: string;
  url?: string;
  size?: string;
  format?: string;
}

export async function searchGoogleDatasetSearch(
  query: string,
  options: { apiKey?: string; cseId?: string; maxResults?: number }
): Promise<DatasetCandidate[]> {
  const { apiKey, cseId, maxResults = 5 } = options;
  if (!apiKey || !cseId) {
    return [];
  }

  const params = new URLSearchParams({
    key: apiKey,
    cx: cseId,
    q: `site:datasetsearch.research.google.com ${query}`,
    num: String(Math.min(maxResults, 10)),
  });

  const res = await fetch(`${CSE_URL}?${params}`);
  if (!res.ok) {
    return [];
  }

  const data = (await res.json()) as {
    items?: Array<{ title?: string; link?: string; snippet?: string }>;
  };
  const items = data.items ?? [];
  return items.map((item, i) => ({
    source: "google_dataset_search",
    name: item.title || item.link || `Result ${i + 1}`,
    slug_or_id: item.link || "",
    description: item.snippet,
    url: item.link,
    format: "unknown",
  }));
}
