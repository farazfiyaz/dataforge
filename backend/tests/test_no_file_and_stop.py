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


def _reply(monkeypatch, text):
    calls = []
    async def fake_llm(messages, tools=None):
        calls.append({"messages": messages, "tools": tools})
        return {"role": "assistant", "content": text}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    return calls


def test_no_file_question_needing_data_says_so(client, monkeypatch):
    calls = _reply(monkeypatch, "NEEDS_FILE")
    res = client.post("/api/agent/", json={"message": "what's the churn rate?"}, headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events] == ["start", "needs_file"]
    assert "no file attached" in events[1][1]["message"]
    assert calls[0]["tools"] is None                    # no code can run without data
    assert "NEEDS_FILE" in calls[0]["messages"][0]["content"]


def test_no_file_general_question_is_answered(client, monkeypatch):
    _reply(monkeypatch, "A p-value is the probability of data at least this extreme if the null is true.")
    res = client.post("/api/agent/", json={"message": "what is a p-value?"}, headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events] == ["start", "final"]
    assert events[1][1]["content"].startswith("A p-value")


def test_no_file_follow_up_keeps_context(client, monkeypatch):
    calls = _reply(monkeypatch, "Answer.")
    client.post("/api/agent/", json={"message": "what is a p-value?", "session_id": "gen"}, headers={"Origin": APP_ORIGIN})
    client.post("/api/agent/", json={"message": "and a t-test?", "session_id": "gen"}, headers={"Origin": APP_ORIGIN})
    assert [m["content"] for m in calls[1]["messages"] if m["role"] == "user"] == ["what is a p-value?", "and a t-test?"]


def test_chat_modes_without_a_file(client, monkeypatch):
    import routers.chat as chat_router
    async def needs(system, message):
        assert "NEEDS_FILE" in system
        return "NEEDS_FILE"
    monkeypatch.setattr(chat_router, "call_ollama", needs)
    res = client.post("/api/chat/", json={"message": "plot my sales", "mode": "plot"}, headers={"Origin": APP_ORIGIN})
    assert res.json()["needs_file"] is True and "no file attached" in res.json()["response"]

    async def general(system, message):
        return "Use df.rolling(7).mean()."
    monkeypatch.setattr(chat_router, "call_ollama", general)
    res = client.post("/api/chat/", json={"message": "moving average code?", "mode": "code"}, headers={"Origin": APP_ORIGIN})
    assert res.json() == {"response": "Use df.rolling(7).mean()."}


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
