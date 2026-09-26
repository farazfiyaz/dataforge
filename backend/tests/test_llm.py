# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Ollama client: proper chat roles, and failures explained in words users can act on."""
import asyncio
import json

import httpx
import pytest

from conftest import APP_ORIGIN
from services import llm


def _mock_ollama(monkeypatch, handler):
    """Route the client's requests to `handler(request) -> httpx.Response` (or raise)."""
    seen = []
    def wrapped(request):
        seen.append(request)
        return handler(request)
    monkeypatch.setattr(llm, "_client",
                        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(wrapped), timeout=timeout))
    return seen


def _reply(content="hello"):
    return lambda request: httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_single_shot_uses_real_system_and_user_roles(monkeypatch):
    seen = _mock_ollama(monkeypatch, _reply("  answer \n"))
    out = asyncio.run(llm.call_ollama("be concise", "what is a median?"))
    assert out == "answer"
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert body["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "what is a median?"},
    ]
    assert "<|system|>" not in seen[0].content.decode()


def _raise(exc):
    def handler(request):
        raise exc
    return handler


@pytest.mark.parametrize("handler, expected", [
    (_raise(httpx.ConnectError("refused")), "ollama serve"),
    (_raise(httpx.ReadTimeout("slow")), "took longer"),
    (lambda r: httpx.Response(404, json={"error": "model 'qwen2.5-coder:7b' not found"}), "ollama pull"),
    (lambda r: httpx.Response(500, text="boom"), "(500)"),
])
def test_failures_become_actionable_messages(monkeypatch, handler, expected):
    _mock_ollama(monkeypatch, handler)
    with pytest.raises(llm.OllamaError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        asyncio.run(llm.chat_ollama([{"role": "user", "content": "hi"}]))


def test_chat_endpoint_shows_the_fix_when_ollama_is_down(client, monkeypatch):
    _mock_ollama(monkeypatch, _raise(httpx.ConnectError("refused")))
    res = client.post("/api/chat/", json={"message": "hi"}, headers={"Origin": APP_ORIGIN})
    assert res.status_code == 503
    assert "ollama serve" in res.json()["detail"]


def test_agent_stream_shows_the_fix_when_model_is_missing(client, monkeypatch):
    _mock_ollama(monkeypatch, lambda r: httpx.Response(404, json={"error": "model 'x' not found"}))
    res = client.post("/api/agent/", json={"message": "hi"}, headers={"Origin": APP_ORIGIN})
    assert "event: error" in res.text
    assert "ollama pull" in res.text
    assert "Ollama error:" not in res.text      # no raw exception prefix
