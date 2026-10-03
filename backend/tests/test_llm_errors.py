# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Ollama failures beyond "not running" and "slow": crashes, out of memory, garbled replies."""
import asyncio

import httpx
import pytest

from conftest import APP_ORIGIN
from services import llm
from test_llm import _mock_ollama, _raise

OOM = ("model requires more system memory (5.6 GiB) than is available (3.1 GiB)")


def _chat():
    return asyncio.run(llm.chat_ollama([{"role": "user", "content": "hi"}], model="qwen2.5-coder:7b"))


def test_out_of_memory_says_to_free_memory(monkeypatch):
    _mock_ollama(monkeypatch, lambda r: httpx.Response(500, json={"error": OOM}))
    with pytest.raises(llm.OllamaError) as e:
        _chat()
    msg = str(e.value)
    assert "enough free memory" in msg
    assert "Close other apps" in msg
    assert "qwen2.5-coder:7b" in msg
    assert "5.6 GiB" in msg                   # Ollama's own numbers are kept
    assert '{"error"' not in msg              # no raw JSON


def test_other_ollama_errors_show_its_message_not_raw_json(monkeypatch):
    _mock_ollama(monkeypatch, lambda r: httpx.Response(500, json={"error": "llama runner process has terminated: exit status 2"}))
    with pytest.raises(llm.OllamaError) as e:
        _chat()
    assert "llama runner process has terminated" in str(e.value)
    assert '{"error"' not in str(e.value)


@pytest.mark.parametrize("exc", [
    httpx.ReadError("connection reset"),
    httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
])
def test_ollama_dropping_the_connection_is_explained(monkeypatch, exc):
    _mock_ollama(monkeypatch, _raise(exc))
    with pytest.raises(llm.OllamaError, match="stopped answering"):
        _chat()


def test_garbled_reply_is_an_ollama_error(monkeypatch):
    _mock_ollama(monkeypatch, lambda r: httpx.Response(200, text="<html>proxy error</html>"))
    with pytest.raises(llm.OllamaError, match="unreadable reply"):
        _chat()


def test_chat_endpoint_reports_crash_as_503_with_the_fix(client, monkeypatch):
    _mock_ollama(monkeypatch, _raise(httpx.ReadError("connection reset")))
    res = client.post("/api/chat/", json={"message": "hi"}, headers={"Origin": APP_ORIGIN})
    assert res.status_code == 503
    assert "stopped answering" in res.json()["detail"]


def test_agent_stream_reports_out_of_memory_in_words(client, monkeypatch):
    _mock_ollama(monkeypatch, lambda r: httpx.Response(500, json={"error": OOM}))
    res = client.post("/api/agent/", json={"message": "hi"}, headers={"Origin": APP_ORIGIN})
    assert "event: error" in res.text
    assert "enough free memory" in res.text
    assert "Ollama error:" not in res.text
