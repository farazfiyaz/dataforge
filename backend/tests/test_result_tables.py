# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Result tables never break a run and are always JSON the agent can stream."""
import json

import pandas as pd
import pytest

import services.executor as executor
from routers.agent import _sse
from services.datastore import put_dataset


@pytest.fixture
def session():
    put_dataset("tables-ds", pd.DataFrame({"x": [1.0, 5.0, 50.0, None], "plan": ["a", "b", "a", "b"]}))
    executor.reset_session("tables")
    return {"session_id": "tables", "dataset_id": "tables-ds"}


def test_binned_column_with_gaps_does_not_break_later_runs(session):
    # pd.cut leaves NaN for 50 (out of range) and None: a categorical with gaps
    out = executor.run_code("df['bucket'] = pd.cut(df['x'], [0, 10, 20])\ndf", **session)
    assert out["error"] is None
    assert [r["bucket"] for r in out["table"]] == ["(0, 10]", "(0, 10]", "", ""]

    # df is the fallback table, so every later run used to fail here
    out = executor.run_code("print(df['x'].mean())", **session)
    assert out["error"] is None
    assert "18.666" in out["stdout"]


def test_date_columns_stream_as_text(session):
    code = ("d = pd.DataFrame({'day': pd.to_datetime(['2024-01-05', None]), 'n': [3, 4]})\n"
            "d")
    out = executor.run_code(code, **session)
    assert out["error"] is None
    assert out["table"] == [{"day": "2024-01-05 00:00:00", "n": 3}, {"day": "", "n": 4}]
    json.loads(_sse("result", {"table": out["table"]}).split("data: ", 1)[1])


def test_numbers_and_text_keep_their_types(session):
    out = executor.run_code("df.groupby('plan', as_index=False)['x'].mean()", **session)
    assert out["table"] == [{"plan": "a", "x": 25.5}, {"plan": "b", "x": 5.0}]


def test_multiindex_columns_become_text_keys(session):
    out = executor.run_code("df.groupby('plan').agg({'x': ['min', 'max']})", **session)
    assert out["error"] is None
    json.dumps(out["table"])
