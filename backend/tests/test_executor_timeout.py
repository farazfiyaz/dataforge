# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Runaway generated code is stopped, and never freezes the server while it runs."""
import asyncio
import time

import pytest

import services.executor as executor
from conftest import APP_ORIGIN


@pytest.fixture
def short_timeout(monkeypatch):
    monkeypatch.setattr(executor, "EXEC_TIMEOUT_S", 0.5)


@pytest.mark.parametrize("code", [
    "while True:\n    pass",
    # generated code often wraps everything in a broad try/except
    "while True:\n    try:\n        x = 1\n    except Exception:\n        pass",
    "def spin():\n    while True:\n        pass\nspin()",
])
def test_infinite_loops_are_stopped(short_timeout, code):
    start = time.monotonic()
    result = executor.run_code(code)
    assert time.monotonic() - start < 3
    assert result["error"].startswith("ExecutionTimeout")


def test_timeout_keeps_output_and_session_state(short_timeout):
    result = executor.run_code("keep = 42\nprint('before')\nwhile True:\n    pass", session_id="t-state")
    assert result["stdout"] == "before\n"
    assert executor.run_code("print(keep)", session_id="t-state")["stdout"].strip() == "42"


def test_trace_hook_is_removed_after_each_run(short_timeout):
    import sys
    executor.run_code("x = 1")
    executor.run_code("while True:\n    pass")
    assert sys.gettrace() is None


def test_normal_code_is_unaffected():
    result = executor.run_code("import numpy as np\nprint(int(np.arange(10).sum()))")
    assert result["error"] is None and result["stdout"].strip() == "45"


def test_event_loop_stays_responsive_while_code_runs(short_timeout):
    async def scenario():
        order = []
        async def ping():
            await asyncio.sleep(0.05)
            order.append("ping")
        async def spin():
            await executor.run_code_async("while True:\n    pass")
            order.append("code")
        await asyncio.gather(spin(), ping())
        return order
    assert asyncio.run(scenario()) == ["ping", "code"]


def test_execute_endpoint_reports_timeout(client, short_timeout):
    res = client.post("/api/execute/", json={"code": "while True:\n    pass"},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 422
    assert "ExecutionTimeout" in res.json()["detail"]
