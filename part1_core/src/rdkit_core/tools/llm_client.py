"""Unified LLM client for NVIDIA Nemotron models.

Uses the OpenAI-compatible API provided by build.nvidia.com or
OpenRouter.  All agents should import this instead of constructing
their own API clients.

Supports:
  - Chat completions (streaming and non-streaming)
  - Structured JSON output via response_format
  - Tool / function calling
  - Automatic retry with exponential backoff
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

# Well-known Nemotron models
NEMOTRON_NANO = "nvidia/nemotron-nano-8b-v1"
NEMOTRON_SUPER_49B = "nvidia/llama-3.3-nemotron-super-49b-v1"
NEMOTRON_SUPER_120B = "nvidia/nemotron-super-120b-a12b"
NEMOTRON_NANO_30B = "nvidia/nemotron-3-nano-30b-a3b"

# Provider base URLs
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """Unified LLM client for Nemotron via OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str = NEMOTRON_SUPER_49B,
        provider: str = "nvidia",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        resolved_key = api_key or os.environ.get(
            "NVIDIA_API_KEY" if provider == "nvidia" else "OPENROUTER_API_KEY",
            "",
        )
        resolved_url = base_url or (
            NVIDIA_BASE_URL if provider == "nvidia" else OPENROUTER_BASE_URL
        )

        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)
        logger.info("llm_client_init", model=model, provider=provider, base_url=resolved_url)

    # ── Core completions ─────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Simple chat completion, returns the assistant message content."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        logger.debug("llm_chat", model=self.model, tokens=response.usage.total_tokens if response.usage else 0)
        return content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat completion that returns parsed JSON.

        Injects a system instruction to force JSON output and attempts
        to parse the response.  Retries on parse failure.
        """
        json_messages = list(messages)
        if json_messages and json_messages[0]["role"] == "system":
            json_messages[0] = {
                **json_messages[0],
                "content": json_messages[0]["content"]
                + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation.",
            }
        else:
            json_messages.insert(0, {
                "role": "system",
                "content": "You MUST respond with valid JSON only. No markdown, no explanation.",
            })

        raw = self.chat(json_messages, temperature=temperature, **kwargs)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return json.loads(cleaned)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Chat completion that returns a validated Pydantic model.

        Sends the JSON schema in the system prompt, parses the response,
        and validates it against the model.
        """
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        structured_messages = list(messages)
        schema_instruction = (
            f"\n\nRespond with a JSON object matching this exact schema:\n{schema_str}"
            "\n\nOutput ONLY valid JSON."
        )

        if structured_messages and structured_messages[0]["role"] == "system":
            structured_messages[0] = {
                **structured_messages[0],
                "content": structured_messages[0]["content"] + schema_instruction,
            }
        else:
            structured_messages.insert(0, {
                "role": "system",
                "content": schema_instruction,
            })

        raw = self.chat(structured_messages, temperature=temperature, **kwargs)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return response_model.model_validate_json(cleaned)

    # ── Tool / function calling ──────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat completion with tool/function calling support.

        Returns the full message object including any tool_calls.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        message = response.choices[0].message
        result: dict[str, Any] = {
            "role": message.role,
            "content": message.content or "",
        }
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return result

    # ── Convenience helpers ──────────────────────────────────────────

    def ask(self, prompt: str, system: str = "") -> str:
        """One-shot question. Simplest interface for quick LLM calls."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)
