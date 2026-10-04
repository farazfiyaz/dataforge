# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""seaborn barplot error bars are described per bar, not as a box plot or stray lines."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from services.chartsummary import describe_figure

PLANS = ["basic", "pro", "enterprise", "team"]


def _data(plans):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"plan": rng.choice(plans, 200), "charge": rng.normal(50, 10, 200)})


def _describe(draw):
    plt.close("all")
    draw()
    fig = plt.gcf()
    fig.savefig(io.BytesIO())
    text = describe_figure(fig)
    plt.close("all")
    return text


@pytest.mark.parametrize("plans", [PLANS, PLANS[:3]])
def test_barplot_error_bars_named_by_bar(plans):
    df = _data(plans)
    text = _describe(lambda: sns.barplot(data=df, x="plan", y="charge"))
    assert "box plot" not in text          # 4 bars used to read as box whiskers
    assert "line" not in text              # 3 bars used to read as unnamed lines
    assert "error bars: " in text
    for plan in plans:
        assert f"{plan} " in text.split("error bars: ")[1]
    means = df.groupby("plan")["charge"].mean()
    for plan in plans:
        assert f"{plan}={means[plan]:.4g}" in text


def test_horizontal_barplot_error_bars():
    df = _data(PLANS)
    text = _describe(lambda: sns.barplot(data=df, x="charge", y="plan"))
    assert "box plot" not in text
    assert "error bars: " in text
    assert all(f"{plan} " in text.split("error bars: ")[1] for plan in PLANS)


def test_error_bar_spans_the_mean():
    df = _data(PLANS)
    text = _describe(lambda: sns.barplot(data=df, x="plan", y="charge", errorbar=("ci", 95)))
    means = df.groupby("plan")["charge"].mean()
    errors = dict(part.rsplit(" ", 1) for part in text.split("error bars: ")[1].split(", "))
    for plan in PLANS:
        lo, hi = (float(v) for v in errors[plan].split("–"))
        assert lo < means[plan] < hi


def test_plain_bars_and_lines_unchanged():
    text = _describe(lambda: (plt.bar(["a", "b"], [1, 2]), plt.plot([0, 1], [1.5, 1.5], label="target")))
    assert "error bars" not in text
    assert "line 'target'" in text


def test_box_plot_still_a_box_plot():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": np.repeat(PLANS, 50), "v": rng.normal(0, 1, 200)})
    assert "box plot" in _describe(lambda: sns.boxplot(data=df, x="g", y="v"))
