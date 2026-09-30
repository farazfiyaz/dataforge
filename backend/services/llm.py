# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import os
import httpx

from services import settings

OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
# Which model answers (and the local default) lives in services/settings.py


async def installed_models() -> list[str]:
    """Models already downloaded into Ollama; empty if Ollama isn't running."""
    try:
        async with _client(5) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
        return sorted(m["name"] for m in resp.json().get("models", []))
    except (httpx.HTTPError, ValueError, KeyError):
        return []


class OllamaError(RuntimeError):
    """An Ollama failure, worded for the user: what went wrong and how to fix it."""


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


async def _post_chat(payload: dict, timeout: float) -> dict:
    """POST to /api/chat, translating transport errors into actionable OllamaErrors."""
    model = payload.get("model")
    try:
        async with _client(timeout) as client:
            resp = await client.post(OLLAMA_CHAT_URL, json=payload)
    except httpx.ConnectError:
        raise OllamaError(
            f"Can't reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve` "
            "(the DataForge desktop app starts it automatically), then try again."
        ) from None
    except httpx.TimeoutException:
        raise OllamaError(
            f"Ollama took longer than {timeout:g}s to answer. The model may still be loading. "
            "Try again, or ask a smaller question."
        ) from None
    if resp.status_code == 404 and "not found" in resp.text.lower():
        raise OllamaError(f"The model `{model}` isn't downloaded yet. Run `ollama pull {model}` and try again.")
    if resp.status_code >= 400:
        raise OllamaError(f"Ollama returned an error ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


async def call_ollama(system_prompt: str, user_message: str, model: str | None = None) -> str:
    """
    Single-shot question → answer text, from the model chosen in Model
    settings: the local Ollama model (default) or the opted-in cloud model.

    Uses /api/chat with real system/user roles, so Ollama applies the model's
    own chat template (the old /api/generate call hand-wrote `<|system|>` tags
    that Qwen doesn't use, so the "system prompt" arrived as user text).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    if settings.provider() == "claude":
        return (await chat_ollama(messages)).get("content", "").strip()
    data = await _post_chat({"model": model or settings.local_model(), "messages": messages, "stream": False},
                            timeout=180)
    return (data.get("message") or {}).get("content", "").strip()


async def chat_ollama(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    """
    Multi-turn chat with native tool calling via Ollama's /api/chat, or via
    the Anthropic API when the user has switched to the cloud model (the
    message shape below is the same either way).

    messages: [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
              Assistant messages may carry "tool_calls"; tool results are sent
              back as {"role": "tool", "content": "<result json>"}.
    tools:    Ollama/OpenAI-style tool definitions, or None for plain chat.

    Returns the assistant message dict:
        {"role": "assistant", "content": "...", "tool_calls": [...]?}
    """
    if settings.provider() == "claude":
        from services import claude   # the cloud path is opt-in; keep it out of local startup
        return await claude.chat(messages, tools, settings.claude_model(), settings.claude_api_key(),
                                 web_search=settings.web_search())

    payload: dict = {
        "model": model or settings.local_model(),
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    data = await _post_chat(payload, timeout=300)
    return data.get("message", {"role": "assistant", "content": ""})
