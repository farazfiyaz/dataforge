# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
pandas 3 regressions: Copy-on-Write made `df[col].fillna(..., inplace=True)`
a silent no-op, and text columns became dtype `str` instead of `object`.
These must hold on pandas 2 and 3 alike.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conftest import APP_ORIGIN
from services.autoplot import generate_all_plots
from services.cleaner import auto_clean
from services.ml import train_model


def _messy_df(n=120):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "plan": rng.choice(["basic", "pro"], n).astype(object),
        "city": rng.choice(["  Sydney", "Perth ", "Hobart"], n).astype(object),
        "amount": rng.choice(["1.5", "2", "3.25"], n).astype(object),
        "churn": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })
    df.loc[:9, "age"] = np.nan
    df.loc[:4, "plan"] = None
    # round-trip through CSV so dtypes are what a real upload produces
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def test_cleaner_actually_fills_the_nulls_it_reports():
    cleaned, report = auto_clean(_messy_df())
    steps = {r["step"] for r in report}
    assert "Filled numeric nulls with column median" in steps
    assert "Filled categorical nulls with column mode" in steps
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["plan"].isna().sum() == 0


def test_cleaner_strips_whitespace_and_converts_numeric_text():
    cleaned, _ = auto_clean(_messy_df())
    assert set(cleaned["city"]) == {"Sydney", "Perth", "Hobart"}
    assert pd.api.types.is_numeric_dtype(cleaned["amount"])


def test_cleaner_never_turns_missing_values_into_nan_text():
    df = pd.DataFrame({"note": ["a", None, " b "] * 10, "x": range(30)})
    df.loc[0, "note"] = None
    cleaned, _ = auto_clean(df)
    assert not (cleaned["note"].astype(object) == "nan").any()


def test_autoplot_includes_text_column_charts():
    df = _messy_df()
    cleaned, _ = auto_clean(df)
    titles = [p["title"] for p in generate_all_plots(df, cleaned)]
    plt.close("all")
    assert "plan Distribution" in titles
    assert "churn Distribution" in titles


def test_train_model_handles_text_labels():
    model = train_model(_messy_df(), target="churn")
    plt.close("all")
    preds = model.predict(_messy_df().drop(columns="churn").head(5))
    assert len(preds) == 5


def test_autoanalyze_endpoint_returns_a_fully_cleaned_csv(client):
    buf = io.BytesIO(_messy_df().to_csv(index=False).encode())
    res = client.post("/api/autoanalyze/", files={"file": ("m.csv", buf, "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    plt.close("all")
    assert res.status_code == 200
    cleaned = pd.read_csv(io.BytesIO(base64.b64decode(res.json()["cleaned_csv_b64"])))
    assert cleaned.isna().sum().sum() == 0
