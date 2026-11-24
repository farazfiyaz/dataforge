# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
train_model's classification report on label sets it used to crash on:
two classes coded 1/2 (roc_curve needs an explicit positive label), and a
rare class that lands only in the training split (the test split looked
binary, so the wrong probability column was scored and the ROC call failed).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from services.ml import train_model


def _df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})


def _tick_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def test_two_classes_coded_one_and_two(capsys):
    df = _df()
    df["grade"] = np.where(df.x > 0, 2, 1)
    train_model(df, target="grade")
    out = capsys.readouterr().out
    assert "ROC AUC: 1.000" in out
    fig = plt.gcf()
    assert len(fig.axes) == 3   # confusion matrix, ROC curve, importances
    assert fig.axes[1].get_title().startswith("ROC curve")
    plt.close("all")


def test_rare_class_missing_from_test_split(capsys):
    df = _df()
    df["tier"] = np.where(df.x > 0, "gold", "silver")
    df.loc[0, "tier"] = "platinum"   # one row, so it can't be stratified
    train_model(df, target="tier")
    out = capsys.readouterr().out
    # three classes: no binary ROC, and no misleading AUC from the wrong column
    assert "ROC AUC" not in out
    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert _tick_labels(fig.axes[0]) == ["gold", "platinum", "silver"]
    plt.close("all")
