# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Auto Analyze's top-value charts: skip identifier columns, include True/False flags."""
import numpy as np
import pandas as pd

from services.autoplot import plot_categoricals
from services.cleaner import auto_clean
from services.loader import read_table


def _titles(df):
    return [p["title"] for p in plot_categoricals(df)]


def _df(n=60):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "order_id": [f"A-{1000 + i}" for i in range(n)],
        "email": [f"user{i}@example.com" for i in range(n)],
        "plan": rng.choice(["basic", "pro", "enterprise"], n),
        "churned": rng.choice([True, False], n),
    })


def test_text_id_columns_get_no_chart():
    titles = _titles(_df())
    assert "order_id Distribution" not in titles
    # unnamed, but every value is unique: a chart of ten bars of height 1
    assert "email Distribution" not in titles
    assert "plan Distribution" in titles


def test_true_false_column_gets_a_chart():
    assert "churned Distribution" in _titles(_df())


def test_true_false_column_from_csv_upload():
    csv = "plan,churned\n" + "".join(f"{p},{c}\n" for p, c in
                                      [("basic", "True"), ("pro", "False"), ("basic", "False")] * 5)
    df, _ = auto_clean(read_table(csv.encode(), "t.csv"))
    assert _titles(df) == ["plan Distribution", "churned Distribution"]


def test_id_columns_do_not_use_up_the_six_chart_limit():
    df = pd.DataFrame({f"id{i}_key": [f"k{i}-{j}" for j in range(30)] for i in range(6)})
    df["segment"] = ["a", "b", "c"] * 10
    assert _titles(df) == ["segment Distribution"]


def test_charts_follow_column_order_and_cap_at_six():
    df = pd.DataFrame({f"c{i}": (["x", "y"] * 10) for i in range(8)})
    df.insert(1, "flag", [True, False] * 10)
    assert _titles(df) == ["c0 Distribution", "flag Distribution",
                           "c1 Distribution", "c2 Distribution",
                           "c3 Distribution", "c4 Distribution"]
