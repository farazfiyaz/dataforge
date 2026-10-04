# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Which column the "Predict …" recommendation picks as the outcome."""
import numpy as np
import pandas as pd

from services.recommend import recommend


def _target(df):
    recs = [r for r in recommend(df, limit=20) if r["id"] == "predict-target"]
    return recs[0]["title"].removeprefix("Predict ") if recs else None


def _bank(n=200, **outcome):
    rng = np.random.default_rng(0)
    cols = {
        "customer_id": range(n),
        "credit_score": rng.integers(400, 850, n),
        "has_cr_card": rng.integers(0, 2, n),
        "is_active_member": rng.integers(0, 2, n),
        "balance": rng.normal(5e4, 1e4, n),
    }
    cols.update({k: v(rng, n) for k, v in outcome.items()})
    return pd.DataFrame(cols)


def test_outcome_name_beats_earlier_flag_features():
    # Used to answer "Predict has_cr_card": the first is_/has_ column won
    df = _bank(churned=lambda rng, n: rng.choice(["yes", "no"], n))
    assert _target(df) == "churned"


def test_last_yes_no_column_beats_earlier_flag_features():
    # `exited` has no outcome-like name; it's the last 0/1 column
    df = _bank(exited=lambda rng, n: rng.integers(0, 2, n))
    assert _target(df) == "exited"


def test_outcome_name_beats_a_later_yes_no_column():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "amount": rng.normal(100, 20, 200),
        "fraud": rng.choice(["yes", "no"], 200),
        "is_weekend": rng.integers(0, 2, 200),
    })
    assert _target(df) == "fraud"


def test_flag_name_still_used_when_nothing_better():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "amount": rng.normal(100, 20, 200),
        "has_plan": rng.choice(["basic", "pro", "none"], 200),
    })
    assert _target(df) == "has_plan"
