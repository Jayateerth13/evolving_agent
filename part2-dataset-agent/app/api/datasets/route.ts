/**
 * Part 2 Dataset Agent API.
 * POST /api/datasets with body: { user_input: string }
 * Returns Part2Output for integration with Part 3, Part 4, and the single app.
 */

import { NextResponse } from "next/server";
import { runDatasetAgent } from "@/lib/flow";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const user_input =
      typeof body?.user_input === "string" ? body.user_input.trim() : "";

    if (!user_input) {
      return NextResponse.json(
        { error: "Missing user_input in request body." },
        { status: 400 }
      );
    }

    const apiKey = process.env.NVIDIA_API_KEY || process.env.NIM_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        {
          error:
            "Missing NVIDIA_API_KEY or NIM_API_KEY. Get one at build.nvidia.com.",
        },
        { status: 503 }
      );
    }

    const result = await runDatasetAgent(user_input, {
      NIM_API_KEY: apiKey,
      KAGGLE_USERNAME: process.env.KAGGLE_USERNAME,
      KAGGLE_KEY: process.env.KAGGLE_KEY,
      GOOGLE_API_KEY: process.env.GOOGLE_API_KEY,
      GOOGLE_CSE_ID: process.env.GOOGLE_CSE_ID,
    });

    return NextResponse.json(result);
  } catch (e) {
    console.error("Part 2 API error:", e);
    return NextResponse.json(
      {
        error: e instanceof Error ? e.message : "Dataset agent failed.",
      },
      { status: 500 }
    );
  }
}
