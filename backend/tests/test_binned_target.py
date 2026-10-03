# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
train_model on a categorical target. "Predict the tenure group" makes the
agent bin a column with pd.cut, whose Interval labels sklearn rejected with
"Unknown label type: unknown"; categoricals with many numeric categories were
treated as regression.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from services.ml import train_model


def _df(n=300):
    rng = np.random.default_rng(1)
    tenure = rng.integers(1, 72, n)
    return pd.DataFrame({
        "tenure": tenure,
        "bill": tenure * 2.0 + rng.normal(0, 5, n),
        "plan": rng.choice(["basic", "pro"], n),
    })


def _confusion_labels():
    fig = plt.gcf()
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    plt.close("all")
    return labels


def test_pd_cut_target_trains_with_bins_in_order(capsys):
    df = _df()
    df["tenure_group"] = pd.cut(df["tenure"], 3)
    model = train_model(df.drop(columns="tenure"), target="tenure_group")
    assert "CLASSIFICATION" in capsys.readouterr().out
    assert len(model.predict(df.drop(columns=["tenure", "tenure_group"]).head(5))) == 5
    labels = _confusion_labels()
    assert labels == [str(c) for c in df["tenure_group"].cat.categories]


def test_named_bins_keep_their_order():
    df = _df()
    df["tier"] = pd.cut(df["tenure"], [0, 12, 36, 72], labels=["new", "mid", "loyal"])
    train_model(df.drop(columns="tenure"), target="tier")
    assert _confusion_labels() == ["new", "mid", "loyal"]


def test_many_numeric_categories_are_classification(capsys):
    df = _df()
    df["bucket"] = (df["tenure"] // 6).astype("category")  # 12 buckets
    train_model(df.drop(columns="tenure"), target="bucket")
    plt.close("all")
    assert "CLASSIFICATION" in capsys.readouterr().out
