# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Model settings: local by default, cloud (Claude) only when opted in."""
import asyncio
import json

import anthropic
import pytest

from conftest import APP_ORIGIN
from services import claude, llm, settings

H = {"Origin": APP_ORIGIN}
KEY = "sk-ant-test-0000000000001234"


# ── Settings API ──

def test_local_is_the_default(client, monkeypatch):
    monkeypatch.setattr("routers.settings.installed_models", _async([]))
    view = client.get("/api/settings/", headers=H).json()
    assert view["provider"] == "local"
    assert view["claude_model"] == "claude-opus-5"
    assert view["has_api_key"] is False


def test_switching_to_cloud_needs_a_key(client, monkeypatch):
    monkeypatch.setattr("routers.settings.installed_models", _async([]))
    res = client.post("/api/settings/", json={"provider": "claude"}, headers=H)
    assert res.status_code == 400 and "API key" in res.json()["detail"]
    assert settings.provider() == "local"


def test_api_key_is_saved_but_never_sent_back(client, monkeypatch):
    monkeypatch.setattr("routers.settings.installed_models", _async([]))
    res = client.post("/api/settings/", json={"provider": "claude", "claude_api_key": KEY,
                                              "claude_model": "claude-sonnet-5"}, headers=H)
    assert res.status_code == 200
    assert KEY not in res.text
    view = res.json()
    assert view["provider"] == "claude" and view["has_api_key"] and view["api_key_hint"] == "…1234"
    assert settings.claude_api_key() == KEY and settings.claude_model() == "claude-sonnet-5"


def test_unknown_values_are_rejected(client):
    assert client.post("/api/settings/", json={"claude_model": "gpt-4"}, headers=H).status_code == 400
    assert client.post("/api/settings/", json={"provider": "other"}, headers=H).status_code == 400


def test_installed_and_suggested_local_models(client, monkeypatch):
    monkeypatch.setattr("routers.settings.installed_models", _async(["qwen2.5-coder:14b", "qwen2.5-coder:7b"]))
    client.post("/api/settings/", json={"local_model": "qwen2.5-coder:14b"}, headers=H)
    view = client.get("/api/settings/", headers=H).json()
    assert view["local_model"] == "qwen2.5-coder:14b"
    assert view["local_models"] == ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]
    assert [m["name"] for m in view["suggested_local_models"]] == ["qwen2.5-coder:32b"]


def test_settings_endpoint_is_protected_from_other_websites(client):
    res = client.post("/api/settings/", json={"provider": "claude", "claude_api_key": KEY},
                      headers={"Origin": "https://evil.example"})
    assert res.status_code == 403


# ── Which model answers ──

def test_local_model_setting_is_used(monkeypatch):
    sent = {}
    async def fake_post(payload, timeout):
        sent.update(payload)
        return {"message": {"role": "assistant", "content": "hi"}}
    monkeypatch.setattr(llm, "_post_chat", fake_post)
    settings.save({"local_model": "qwen2.5-coder:14b"})
    asyncio.run(llm.chat_ollama([{"role": "user", "content": "x"}]))
    assert sent["model"] == "qwen2.5-coder:14b"


def test_cloud_setting_routes_to_claude(monkeypatch):
    seen = {}
    async def fake_claude(messages, tools, model, api_key, web_search=False):
        seen.update(model=model, api_key=api_key, tools=tools)
        return {"role": "assistant", "content": "from claude"}
    async def no_ollama(*a, **k):
        raise AssertionError("Ollama must not be called in cloud mode")
    monkeypatch.setattr(claude, "chat", fake_claude)
    monkeypatch.setattr(llm, "_post_chat", no_ollama)
    settings.save({"provider": "claude", "claude_api_key": KEY})
    assert asyncio.run(llm.call_ollama("sys", "question")) == "from claude"
    assert seen == {"model": "claude-opus-5", "api_key": KEY, "tools": None}


# ── Translating the conversation to and from the Messages API ──

TOOL = {"type": "function", "function": {"name": "run_python", "description": "Run code",
                                         "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}


def test_tool_definition_translation():
    assert claude.to_claude_tool(TOOL) == {"name": "run_python", "description": "Run code",
                                           "input_schema": TOOL["function"]["parameters"]}


def test_history_translation_pairs_calls_and_results():
    history = [
        {"role": "system", "content": "You are DataForge."},
        {"role": "user", "content": "churn rate?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "run_python", "arguments": {"code": "a"}}},
            {"function": {"name": "run_python", "arguments": '{"code": "b"}'}}]},
        {"role": "tool", "content": json.dumps({"stdout": "1"})},
        {"role": "tool", "content": json.dumps({"error": "boom"})},
        {"role": "assistant", "content": "It's 26%."},
    ]
    system, msgs = claude.to_claude_messages(history)
    assert system == "You are DataForge."
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    ids = [b["id"] for b in msgs[1]["content"]]
    results = msgs[2]["content"]                          # both results in ONE user message
    assert [r["tool_use_id"] for r in results] == ids
    assert msgs[1]["content"][1]["input"] == {"code": "b"}
    assert "is_error" not in results[0] and results[1]["is_error"] is True


def test_raw_claude_blocks_are_echoed_back_unchanged():
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"},
           {"type": "tool_use", "id": "toolu_abc", "name": "run_python", "input": {"code": "x"}}]
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": "", "claude_content": raw,
                "tool_calls": [{"function": {"name": "run_python", "arguments": {"code": "x"}}}]},
               {"role": "tool", "content": "{}"}]
    _, msgs = claude.to_claude_messages(history)
    assert msgs[1]["content"] == raw
    assert msgs[2]["content"][0]["tool_use_id"] == "toolu_abc"


def _message(content, stop_reason):
    return anthropic.types.Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}})


def test_reply_translation_tool_use():
    msg = claude.from_claude_response(_message(
        [{"type": "text", "text": "Let me check."},
         {"type": "tool_use", "id": "toolu_1", "name": "run_python", "input": {"code": "df.head()"}}],
        "tool_use"))
    assert msg["content"] == "Let me check."
    assert msg["tool_calls"] == [{"id": "toolu_1", "function": {"name": "run_python",
                                                                "arguments": {"code": "df.head()"}}}]
    assert [b["type"] for b in msg["claude_content"]] == ["text", "tool_use"]


def test_reply_translation_refusal_and_cutoff():
    assert "declined" in claude.from_claude_response(_message([], "refusal"))["content"]
    cut = claude.from_claude_response(_message([{"type": "text", "text": "partial"}], "max_tokens"))
    assert "cut off" in cut["content"]


def test_bad_key_gives_a_clear_message():
    # A real request with a fake key: proves the SDK accepts the request we build
    # (model, tools, fallback beta) and that a 401 is explained in words
    with pytest.raises(llm.OllamaError, match="API key was rejected"):
        asyncio.run(claude.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
                                [TOOL], "claude-opus-5", "sk-ant-invalid-key-for-tests"))


def _async(value):
    async def f(*a, **k):
        return value
    return f
