# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
A run_python `code` argument wrapped in a markdown fence is run as the code
inside it, not as a SyntaxError on line 1.
"""
import json

import pytest

from conftest import APP_ORIGIN
from routers import agent as agent_router
from routers.agent import _normalize_calls, _unfence

FENCE = "`" * 3
CODE = "x = 6 * 7\nprint(x)"


@pytest.mark.parametrize("wrapped", [
    f"{FENCE}python\n{CODE}\n{FENCE}",
    f"{FENCE}py\n{CODE}\n{FENCE}\n",
    f"{FENCE}\n{CODE}\n{FENCE}",
    f"Here is the code:\n{FENCE}python\n{CODE}\n{FENCE}\nThis prints 42.",
    f"{FENCE}python\n{CODE}",                                   # closing fence cut off
])
def test_unfences_wrapped_code(wrapped):
    assert _unfence(wrapped) == CODE


@pytest.mark.parametrize("code", [
    CODE,
    f'print("{FENCE}python")',                                  # a fence inside a string is just text
    "def f(:\n    pass",                                        # broken, but no fence: left for the traceback
])
def test_leaves_other_code_alone(code):
    assert _unfence(code) == code


def test_normalize_unfences_native_and_string_arguments():
    native = [{"function": {"name": "run_python", "arguments": {"code": f"{FENCE}python\n{CODE}\n{FENCE}"}}}]
    as_string = [{"function": {"name": "run_python",
                               "arguments": json.dumps({"code": f"{FENCE}python\n{CODE}\n{FENCE}"})}}]
    for calls in (native, as_string):
        assert _normalize_calls(calls)[0]["function"]["arguments"]["code"] == CODE


def test_agent_runs_fenced_code_argument(client, monkeypatch):
    replies = iter([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "run_python", "arguments": {"code": f"{FENCE}python\n{CODE}\n{FENCE}"}}}]},
        {"role": "assistant", "content": "The answer is 42."},
    ])
    async def fake_llm(messages, tools=None):
        return next(replies)
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)

    res = client.post("/api/agent/", json={"message": "what is 6*7?"}, headers={"Origin": APP_ORIGIN})
    events = [(b.split("\n")[0][7:], json.loads(b.split("\n", 1)[1][6:])) for b in res.text.strip().split("\n\n")]
    assert [e for e, _ in events] == ["start", "code", "result", "final"]
    assert events[1][1]["code"] == CODE              # the user sees clean code too
    assert events[2][1]["error"] is None
    assert events[2][1]["stdout"].strip() == "42"
