/**
 * Kaggle dataset search via REST API.
 * Requires KAGGLE_USERNAME and KAGGLE_KEY (from kaggle.json or account API token).
 */

const KAGGLE_LIST = "https://www.kaggle.com/api/v1/datasets/list";

export interface KaggleDatasetItem {
  ref: string; // "username/dataset-slug"
  title: string;
  description?: string;
  size?: string;
  downloadCount?: number;
}

export interface DatasetCandidate {
  source: string;
  name: string;
  slug_or_id: string;
  description?: string;
  url?: string;
  size?: string;
  format?: string;
}

function authHeader(username: string, key: string): string {
  const encoded = Buffer.from(`${username}:${key}`, "utf8").toString("base64");
  return `Basic ${encoded}`;
}

export async function searchKaggle(
  query: string,
  options: { username: string; key: string; maxResults?: number }
): Promise<DatasetCandidate[]> {
  const { username, key, maxResults = 10 } = options;
  const url = new URL(KAGGLE_LIST);
  url.searchParams.set("group", "general");
  url.searchParams.set("sort", "relevance");
  url.searchParams.set("search", query);
  url.searchParams.set("page", "1");
  url.searchParams.set("pageSize", String(maxResults));

  const res = await fetch(url.toString(), {
    headers: {
      Authorization: authHeader(username, key),
      "Content-Type": "application/json",
    },
  });

  if (!res.ok) {
    if (res.status === 401) {
      throw new Error("Kaggle: Invalid credentials. Set KAGGLE_USERNAME and KAGGLE_KEY.");
    }
    throw new Error(`Kaggle API error ${res.status}: ${await res.text()}`);
  }

  const data = (await res.json()) as KaggleDatasetItem[];
  return (Array.isArray(data) ? data : []).map((item) => ({
    source: "kaggle",
    name: item.title || item.ref,
    slug_or_id: item.ref,
    description: item.description,
    url: `https://www.kaggle.com/datasets/${item.ref}`,
    size: item.size,
    format: "csv",
  }));
}
