# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Next-step recommendations: from the data, on upload, and after agent answers."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from conftest import APP_ORIGIN
from routers import agent as agent_router
from services.recommend import recommend


def _churn_df(n=200):
    rng = np.random.default_rng(0)
    tenure = rng.integers(1, 60, n)
    return pd.DataFrame({
        "customer_id": range(1000, 1000 + n),
        "tenure_months": tenure,
        "total_spend": tenure * 50 + rng.normal(0, 5, n),
        "plan": rng.choice(["basic", "pro", "enterprise"], n),
        "churn": rng.choice(["yes", "no"], n, p=[0.1, 0.9]),
    })


def _ids(recs):
    return [r["id"] for r in recs]


def test_recommends_predicting_the_outcome_column_first():
    recs = recommend(_churn_df())
    assert recs[0]["id"] == "predict-target"
    assert "`churn`" in recs[0]["prompt"]
    assert "customer_id" in recs[0]["prompt"]        # told to exclude the identifier
    assert "imbalanced" in recs[0]["reason"]
    assert "segment-target" in _ids(recs)


def test_every_recommendation_is_ready_to_send():
    for r in recommend(_churn_df(), limit=20):
        assert set(r) == {"id", "title", "reason", "prompt"}
        assert r["title"] and r["reason"] and r["prompt"]


def test_two_valued_grouping_column_is_not_a_prediction_target():
    df = pd.DataFrame({"region": ["N", "S"] * 50, "sales": np.arange(100.0)})
    assert "predict-target" not in _ids(recommend(df, limit=20))


def test_flags_missing_values_duplicates_correlation_and_dates():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=100).astype(str),
        "sales": rng.random(100) * 100,
        "cost": rng.random(100),
    })
    df["profit"] = df["sales"] * 0.3
    df.loc[:40, "cost"] = None
    df = pd.concat([df, df.head(3)])
    ids = _ids(recommend(df, limit=20))
    assert {"missing-values", "duplicates", "correlation", "trend"} <= set(ids)


def test_already_asked_prompts_are_not_repeated():
    df = _churn_df()
    first = recommend(df)[0]
    again = recommend(df, asked=[first["prompt"]])
    assert first["id"] not in _ids(again)


@pytest.mark.parametrize("df", [pd.DataFrame(), pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [None, None]})])
def test_degenerate_data_does_not_crash(df):
    recommend(df)


def test_upload_returns_recommendations(client):
    buf = io.BytesIO(_churn_df().to_csv(index=False).encode())
    res = client.post("/api/upload/", files={"file": ("churn.csv", buf, "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 200
    assert res.json()["profile"]["recommendations"][0]["id"] == "predict-target"


def _events(res):
    out = []
    for block in res.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _upload(client):
    buf = io.BytesIO(_churn_df().to_csv(index=False).encode())
    return client.post("/api/upload/", files={"file": ("churn.csv", buf, "text/csv")},
                       headers={"Origin": APP_ORIGIN}).json()["dataset_id"]


def test_agent_suggests_next_steps_after_answering(client, monkeypatch):
    async def fake_llm(messages, tools=None):
        return {"role": "assistant", "content": "Churn is 10%."}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)

    res = client.post("/api/agent/", json={"message": "what is the churn rate?", "dataset_id": _upload(client)},
                      headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events] == ["start", "final", "recommendations"]
    recs = events[-1][1]
    assert recs["stuck"] is False and 1 <= len(recs["items"]) <= 3


def test_agent_offers_easier_questions_when_stuck(client, monkeypatch):
    async def looping_llm(messages, tools=None):   # never produces a final answer
        return {"role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "run_python", "arguments": {"code": "x = 1"}}}]}
    monkeypatch.setattr(agent_router, "chat_ollama", looping_llm)

    res = client.post("/api/agent/", json={"message": "do everything", "dataset_id": _upload(client)},
                      headers={"Origin": APP_ORIGIN})
    kinds = [e for e, _ in _events(res)]
    assert kinds[-2:] == ["final", "recommendations"]
    assert _events(res)[-1][1]["stuck"] is True


def test_agent_without_data_sends_no_recommendations(client, monkeypatch):
    async def fake_llm(messages, tools=None):
        return {"role": "assistant", "content": "Hi!"}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)

    res = client.post("/api/agent/", json={"message": "hello"}, headers={"Origin": APP_ORIGIN})
    assert [e for e, _ in _events(res)] == ["start", "final"]
