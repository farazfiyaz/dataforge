# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Date columns are profiled as dates, not as categories with "top values"."""
import io

import numpy as np
import pandas as pd

from services.eda import run_eda


def _kinds(profile):
    return {c["name"]: c for c in profile["columns"]}


def test_text_dates_from_csv_are_datetime():
    df = pd.DataFrame({
        "signup_date": ["2024-01-05", "2024-03-01", None, "2024-02-10"],
        "plan": ["basic", "pro", "pro", "basic"],
        "amount": [1.0, 2.0, 3.0, 4.0],
    })
    cols = _kinds(run_eda(df))
    d = cols["signup_date"]
    assert d["kind"] == "datetime"
    assert d["min"].startswith("2024-01-05") and d["max"].startswith("2024-03-01")
    assert d["null_count"] == 1
    assert cols["plan"]["kind"] == "categorical"


def test_excel_datetime_column_with_blanks(client):
    df = pd.DataFrame({
        "signup": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
        "plan": ["a", None, "b"],
        "amt": [1.5, np.nan, 3.0],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    res = client.post("/api/upload/", files={"file": ("t.xlsx", buf.getvalue())})
    assert res.status_code == 200
    cols = _kinds(res.json()["profile"])
    assert cols["signup"]["kind"] == "datetime"
    assert cols["signup"]["max"].startswith("2024-03-01")
    assert cols["plan"]["kind"] == "categorical"


def test_all_missing_dates_have_no_range():
    df = pd.DataFrame({"closed_at": pd.to_datetime(pd.Series([None, None])), "x": [1, 2]})
    d = _kinds(run_eda(df))["closed_at"]
    assert d["kind"] == "datetime" and d["min"] is None and d["max"] is None


def test_text_named_like_a_date_but_not_dates_stays_categorical():
    df = pd.DataFrame({"day_type": ["weekday", "weekend", "weekday"]})
    assert _kinds(run_eda(df))["day_type"]["kind"] == "categorical"
