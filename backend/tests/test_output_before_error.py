# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Output printed before an error is kept, for the user and for the agent."""
import json

import pandas as pd
import pytest

import services.executor as executor
from routers.agent import _tool_result_for_llm
from services.datastore import put_dataset


@pytest.fixture
def session():
    put_dataset("partial-ds", pd.DataFrame({"plan": ["basic", "pro"] * 5, "age": range(10)}))
    executor.reset_session("partial")
    return {"session_id": "partial", "dataset_id": "partial-ds"}


def test_stdout_before_error_is_kept(session):
    out = executor.run_code("print(df.shape)\nprint(sorted(df.columns))\ndf['nope']", **session)
    assert "KeyError" in out["error"]
    assert "(10, 2)" in out["stdout"] and "'plan'" in out["stdout"]


def test_error_without_output_has_empty_stdout(session):
    out = executor.run_code("1 / 0", **session)
    assert "ZeroDivisionError" in out["error"] and out["stdout"] == ""


def test_agent_sees_output_before_error():
    payload = json.loads(_tool_result_for_llm({"stdout": "(10, 2)\n", "error": "KeyError: 'nope'"}))
    assert payload["error"] == "KeyError: 'nope'"
    assert payload["stdout_before_error"] == "(10, 2)\n"


def test_agent_error_payload_unchanged_without_output():
    assert json.loads(_tool_result_for_llm({"stdout": "", "error": "boom"})) == {"error": "boom"}
