# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Opt-in web search: off by default; local model gets a web_search tool; Claude uses its own."""
import json

import anthropic

from conftest import APP_ORIGIN
from routers import agent as agent_router
from services import claude, settings, websearch

H = {"Origin": APP_ORIGIN}

DDG_PAGE = (
    '<div class="result"><a rel="nofollow" class="result__a" '
    'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FP-value&amp;rut=x">'
    'p-value - <b>Wikipedia</b></a>\n'
    '<a class="result__snippet" href="#">In null-hypothesis testing, the <b>p-value</b> is&hellip;</a></div>\n'
    '<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Sponsored</a>\n'
    '<a class="result__snippet" href="#">ad</a></div>\n'
    '<div class="result"><a class="result__a" href="https://www.statology.org/p-values/">p-values Explained</a>\n'
    '<a class="result__snippet" href="#">Plain English.</a></div>\n'
)


def _events(res):
    return [(b.split("\n")[0][7:], json.loads(b.split("\n", 1)[1][6:])) for b in res.text.strip().split("\n\n")]


def _call(query):
    # qwen2.5-coder writes tool calls as text; they're recovered like run_python calls
    return {"role": "assistant", "content": json.dumps({"name": "web_search", "arguments": {"query": query}})}


def test_parses_results_unwraps_links_and_skips_ads():
    assert websearch.parse_results(DDG_PAGE) == [
        {"title": "p-value - Wikipedia", "url": "https://en.wikipedia.org/wiki/P-value",
         "snippet": "In null-hypothesis testing, the p-value is…"},
        {"title": "p-values Explained", "url": "https://www.statology.org/p-values/", "snippet": "Plain English."},
    ]


def test_off_by_default_so_no_search_tool(client, monkeypatch):
    seen = []
    async def fake_llm(messages, tools=None):
        seen.append(tools)
        return {"role": "assistant", "content": "A p-value is..."}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    client.post("/api/agent/", json={"message": "what is a p-value?"}, headers=H)
    assert settings.web_search() is False and seen == [None]


def test_local_model_searches_then_answers(client, monkeypatch):
    settings.save({"web_search": True})
    searched = []
    async def fake_search(query, max_results=5):
        searched.append(query)
        return [{"title": "p-value - Wikipedia", "url": "https://en.wikipedia.org/wiki/P-value", "snippet": "..."}]
    monkeypatch.setattr(websearch, "search", fake_search)
    replies = iter([_call("p-value definition"),
                    {"role": "assistant", "content": "A p-value is... (source: en.wikipedia.org/wiki/P-value)"}])
    tool_sets = []
    async def fake_llm(messages, tools=None):
        tool_sets.append([t["function"]["name"] for t in tools or []])
        return next(replies)
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)

    events = _events(client.post("/api/agent/", json={"message": "what is a p-value?"}, headers=H))
    assert [e for e, _ in events] == ["start", "search", "final"]
    assert searched == ["p-value definition"]
    assert events[1][1]["results"][0]["url"] == "https://en.wikipedia.org/wiki/P-value"
    assert tool_sets[0] == ["web_search"]


def test_endless_searching_is_cut_off_with_an_answer(client, monkeypatch):
    settings.save({"web_search": True})
    async def fake_search(query, max_results=5):
        return []
    monkeypatch.setattr(websearch, "search", fake_search)
    calls = []
    async def fake_llm(messages, tools=None):
        calls.append(tools)
        return _call("x") if tools else {"role": "assistant", "content": "Best answer I have."}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    events = _events(client.post("/api/agent/", json={"message": "q"}, headers=H))
    assert events[-1] == ("final", {"content": "Best answer I have.", "steps": 0, "sources": None})
    assert calls[-1] is None and len(calls) == agent_router.MAX_SEARCH_ROUNDS + 1


def test_search_failure_is_told_to_the_model_not_a_crash(client, monkeypatch):
    settings.save({"web_search": True})
    async def offline(query, max_results=5):
        raise websearch.WebSearchError("Web search isn't reachable right now (no internet connection?).")
    monkeypatch.setattr(websearch, "search", offline)
    replies = iter([_call("x"), {"role": "assistant", "content": "Answer from memory."}])
    async def fake_llm(messages, tools=None):
        return next(replies)
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    events = _events(client.post("/api/agent/", json={"message": "q"}, headers=H))
    assert events[1][0] == "search" and "isn't reachable" in events[1][1]["error"]
    assert events[-1][1]["content"] == "Answer from memory."


def test_data_session_can_search_too(client, monkeypatch):
    settings.save({"web_search": True})
    async def fake_search(query, max_results=5):
        return [{"title": "t", "url": "https://example.com", "snippet": "s"}]
    monkeypatch.setattr(websearch, "search", fake_search)
    replies = iter([_call("industry churn benchmark"), {"role": "assistant", "content": "Yours is above average."}])
    async def fake_llm(messages, tools=None):
        return next(replies)
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    events = _events(client.post("/api/agent/", json={"message": "is our churn high?", "csv_data": "a\n1\n"},
                                 headers=H))
    assert [e for e, _ in events][:3] == ["start", "search", "final"]


def test_cloud_mode_uses_claudes_own_search_tool():
    assert claude._web_search_tool("claude-opus-5")["type"] == "web_search_20260209"
    assert claude._web_search_tool("claude-haiku-4-5")["type"] == "web_search_20250305"


def test_cloud_sources_are_listed():
    msg = anthropic.types.Message.model_validate({
        "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5", "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "p-value"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                {"type": "web_search_result", "url": "https://en.wikipedia.org/wiki/P-value",
                 "title": "p-value", "encrypted_content": "x", "page_age": None}]},
            {"type": "text", "text": "A p-value is..."}]})
    reply = claude.from_claude_response(msg)
    assert reply["content"] == "A p-value is..."
    assert reply["sources"] == [{"title": "p-value", "url": "https://en.wikipedia.org/wiki/P-value"}]
    assert [b["type"] for b in reply["claude_content"]] == ["server_tool_use", "web_search_tool_result", "text"]


def test_setting_round_trips(client, monkeypatch):
    async def none():
        return []
    monkeypatch.setattr("routers.settings.installed_models", none)
    assert client.post("/api/settings/", json={"web_search": True}, headers=H).json()["web_search"] is True
