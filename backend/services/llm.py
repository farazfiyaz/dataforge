# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import json
import httpx

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL   = "qwen2.5-coder:7b"


async def call_ollama(system_prompt: str, user_message: str) -> str:
    """
    Send a request to Ollama and return the full response as a string.
    Uses stream=False so Electron's network stack doesn't choke on chunked encoding.
    """
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>",
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


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

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(OLLAMA_CHAT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {"role": "assistant", "content": ""})
