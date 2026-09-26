# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""The sandbox shows a trailing expression's value, like a notebook cell."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import services.executor as executor
from services.datastore import put_dataset
from services.ml import train_model


@pytest.fixture
def churn_session():
    df = pd.DataFrame({"plan": ["a", "b", "a", "b"] * 30, "age": range(120),
                       "churn": ["yes", "no", "no", "no"] * 30})
    put_dataset("echo-ds", df)
    executor.reset_session("echo")
    return {"session_id": "echo", "dataset_id": "echo-ds"}


def test_trailing_expression_is_shown(churn_session):
    out = executor.run_code("df['churn'].value_counts()", **churn_session)
    assert "no" in out["stdout"] and "90" in out["stdout"]


def test_trailing_dataframe_becomes_the_table(churn_session):
    out = executor.run_code("summary = df.describe()\ndf.head(3)", **churn_session)
    assert len(out["table"]) == 3            # the head the code ended on, not `summary`
    assert "plan" in out["stdout"]


def test_statements_and_none_are_not_echoed(churn_session):
    assert executor.run_code("x = 5", **churn_session)["stdout"] == ""
    assert executor.run_code("print('hi')", **churn_session)["stdout"] == "hi\n"   # print() returns None


def test_plot_return_values_are_not_echoed(churn_session):
    out = executor.run_code("plt.plot([1, 2, 3])[0]", **churn_session)
    assert out["stdout"] == "" and out["charts"]


def test_long_values_are_truncated(churn_session):
    out = executor.run_code("'x' * 10000", **churn_session)
    assert "(truncated)" in out["stdout"] and len(out["stdout"]) < 4100


def test_syntax_errors_still_reported(churn_session):
    assert "SyntaxError" in executor.run_code("df.head(", **churn_session)["error"]


def test_train_model_prints_feature_importances(capsys):
    df = pd.DataFrame({"signal": list(range(100)) * 2, "noise": [1, 2] * 100,
                       "y": ["a"] * 100 + ["b"] * 100})
    df["signal"] = [i if y == "a" else i + 1000 for i, y in zip(df["signal"], df["y"])]
    train_model(df, target="y")
    plt.close("all")
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("Feature importance"))
    assert line.index("signal") < line.index("noise")
