# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Identifier columns (customer_id, row counters) are flagged and kept out of charts and models."""
import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from services.autoplot import _numeric_columns
from services.eda import find_date_column, is_id_column, run_eda
from services.ml import train_model


def _df(n=150):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "customer_id": np.arange(5000, 5000 + n),
        "order_uuid": [f"u{i}" for i in range(n)],
        "age": rng.integers(18, 80, n),
        "spend": rng.random(n) * 100,
        "signup_date": pd.date_range("2024-01-01", periods=n).astype(str),
        "churn": rng.choice(["yes", "no"], n),
    })


@pytest.mark.parametrize("col, expected", [
    ("customer_id", True), ("order_uuid", True),
    ("age", False), ("spend", False), ("churn", False),
])
def test_is_id_column(col, expected):
    df = _df()
    assert is_id_column(df[col], col) is expected


def test_unnamed_sequential_counter_is_an_id():
    s = pd.Series(range(100))
    assert is_id_column(s, "row")


def test_profile_flags_ids_and_dates():
    profile = run_eda(_df())
    flags = {c["name"]: c["is_id"] for c in profile["columns"]}
    assert flags["customer_id"] and not flags["age"]
    assert profile["date_column"] == "signup_date"


def test_no_date_column_when_there_is_none():
    assert find_date_column(pd.DataFrame({"a": [1, 2]})) is None


def test_autoplot_skips_id_columns():
    assert _numeric_columns(_df(), 12) == ["age", "spend"]


def test_train_model_drops_id_columns_unless_asked():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model = train_model(_df(), target="churn")
    plt.close("all")
    used = model.named_steps["pre"].transformers_[0][2]
    assert "customer_id" not in used
    assert "customer_id" in out.getvalue()      # reported as dropped

    with contextlib.redirect_stdout(io.StringIO()):
        model = train_model(_df(), target="churn", features=["customer_id", "age"])
    plt.close("all")
    assert "customer_id" in model.named_steps["pre"].transformers_[0][2]


def test_sorted_unique_feature_is_not_mistaken_for_an_id():
    # unique and increasing, but not a row counter, e.g. data sorted by price
    s = pd.Series([3, 7, 8, 15, 40, 41, 90] * 1 + list(range(100, 200, 3)))
    assert not is_id_column(s, "price")
