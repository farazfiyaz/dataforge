# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Grouped bars and pie charts are described by category, not by bar position."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from services.chartsummary import describe_figure

DF = pd.DataFrame({
    "plan": ["basic"] * 3 + ["pro"] * 2 + ["enterprise"],
    "churn": ["yes", "yes", "no", "no", "yes", "no"],
})
COUNTS = DF.groupby(["plan", "churn"]).size().unstack(fill_value=0)
GROUPED = "bars 'no': basic=1, enterprise=1, pro=1; bars 'yes': basic=2, enterprise=0, pro=1"


def _describe(draw):
    plt.close("all")
    draw()
    fig = plt.gcf()
    fig.savefig(io.BytesIO())
    text = describe_figure(fig)
    plt.close("all")
    return text


@pytest.mark.parametrize("draw", [
    lambda: COUNTS.plot(kind="bar"),                  # dodged: bars sit beside each tick
    lambda: COUNTS.plot(kind="barh", stacked=True),
])
def test_pandas_grouped_bars_are_named_by_category(draw):
    assert GROUPED in _describe(draw)


def test_seaborn_hue_bars_take_series_names_from_the_legend():
    text = _describe(lambda: sns.countplot(data=DF, x="plan", hue="churn",
                                           order=["basic", "enterprise", "pro"], hue_order=["no", "yes"]))
    # seaborn skips the empty enterprise/yes bar
    assert "bars 'no': basic=1, enterprise=1, pro=1; bars 'yes': basic=2, pro=1" in text


def test_histogram_bars_are_not_snapped_to_ticks():
    text = _describe(lambda: plt.hist(np.random.default_rng(0).normal(5, 2, 500)))
    assert "bars: -2.102=2, -0.7092=3" in text


@pytest.mark.parametrize("draw, expected", [
    (lambda: DF["plan"].value_counts().plot.pie(autopct="%1.1f%%"),
     "pie, 3 slices: basic 50.0%, pro 33.3%, enterprise 16.7%"),
    (lambda: plt.pie([3, 5, 2], labels=["a", "b", "c"]), "pie, 3 slices: a 30.0%, b 50.0%, c 20.0%"),
    (lambda: plt.pie([1, 3]), "pie, 2 slices: slice 1 25.0%, slice 2 75.0%"),
])
def test_pie_charts_report_each_slice_share(draw, expected):
    assert expected in _describe(draw)
