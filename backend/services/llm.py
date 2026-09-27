# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import os
import httpx

OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
# Electron picks the model size that fits the machine's RAM (see electron/main.js
# pickModel()) and passes it in via this env var; falls back to the 7B model when
# run outside Electron (e.g. `python app.py` directly).
DEFAULT_MODEL   = os.environ.get("DATAFORGE_MODEL", "qwen2.5-coder:7b")


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


async def call_ollama(system_prompt: str, user_message: str, model: str = DEFAULT_MODEL) -> str:
    """
    Single-shot question → answer text.

    Uses /api/chat with real system/user roles, so Ollama applies the model's
    own chat template (the old /api/generate call hand-wrote `<|system|>` tags
    that Qwen doesn't use, so the "system prompt" arrived as user text).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    data = await _post_chat({"model": model, "messages": messages, "stream": False}, timeout=180)
    return (data.get("message") or {}).get("content", "").strip()


async def chat_ollama(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Multi-turn chat with native tool calling via Ollama's /api/chat.

    messages: [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
              Assistant messages may carry "tool_calls"; tool results are sent
              back as {"role": "tool", "content": "<result json>"}.
    tools:    Ollama/OpenAI-style tool definitions, or None for plain chat.

    Returns the assistant message dict:
        {"role": "assistant", "content": "...", "tool_calls": [...]?}
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    data = await _post_chat(payload, timeout=300)
    return data.get("message", {"role": "assistant", "content": ""})
