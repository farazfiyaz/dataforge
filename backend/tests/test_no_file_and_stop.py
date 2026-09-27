# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""No data → a clear "attach a file" reply; Stop → the half-finished turn is forgotten."""
import asyncio
import json

from conftest import APP_ORIGIN
from routers import agent as agent_router
from services import workspace


def _events(res):
    return [(b.split("\n")[0][7:], json.loads(b.split("\n", 1)[1][6:])) for b in res.text.strip().split("\n\n")]


def _no_llm(monkeypatch):
    calls = []
    async def fake_llm(messages, tools=None):
        calls.append(messages)
        return {"role": "assistant", "content": "should not be called"}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    return calls


def test_no_file_says_so_without_calling_the_model(client, monkeypatch):
    calls = _no_llm(monkeypatch)
    res = client.post("/api/agent/", json={"message": "what's the churn rate?"}, headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events] == ["start", "needs_file"]
    assert "no file attached" in events[1][1]["message"]
    assert calls == []


def test_file_lost_after_restart_is_explained(client, monkeypatch):
    _no_llm(monkeypatch)
    res = client.post("/api/agent/", json={"message": "go", "dataset_id": "gone-after-restart"},
                      headers={"Origin": APP_ORIGIN})
    assert "isn't loaded any more" in _events(res)[1][1]["message"]


def test_workspace_folder_counts_as_data(client, monkeypatch, tmp_path):
    calls = _no_llm(monkeypatch)
    workspace.set_workspace(str(tmp_path))
    res = client.post("/api/agent/", json={"message": "list my files"}, headers={"Origin": APP_ORIGIN})
    assert "needs_file" not in res.text
    assert len(calls) == 1


def test_stopping_mid_run_forgets_the_unfinished_turn(monkeypatch):
    session = "stop-test"
    history = [{"role": "system", "content": "sys"},
               {"role": "user", "content": "earlier question"},
               {"role": "assistant", "content": "earlier answer"}]
    agent_router._CONVERSATIONS[session] = history

    async def slow_llm(messages, tools=None):
        await asyncio.sleep(10)    # the user presses Stop while the model is thinking
    monkeypatch.setattr(agent_router, "chat_ollama", slow_llm)

    async def scenario():
        req = agent_router.AgentRequest(message="new question", csv_data="a\n1\n", session_id=session)
        response = await agent_router.agent(req)
        body = response.body_iterator
        first = await body.__anext__()                 # "start" arrives…
        pending = asyncio.ensure_future(body.__anext__())
        await asyncio.sleep(0.05)
        pending.cancel()                               # …then the browser disconnects
        try:
            await pending
        except asyncio.CancelledError:
            pass
        await body.aclose()
        return first

    assert "event: start" in asyncio.run(scenario())
    assert [m["content"] for m in agent_router._CONVERSATIONS[session]] == \
        ["sys", "earlier question", "earlier answer"]
