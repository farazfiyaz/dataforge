# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Violin, strip and swarm plots are described per category, not as raw shapes."""
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import services.executor as executor
from routers.agent import _tool_result_for_llm
from services.chartsummary import describe_figure

rng = np.random.default_rng(0)
DF = pd.DataFrame({
    "plan": np.repeat(["basic", "pro", "team"], 40),
    "spend": np.r_[rng.normal(20, 5, 40), rng.normal(50, 8, 40), rng.normal(80, 10, 40)],
    "seg": np.tile(["a", "b"], 60),
})
MEDIANS = DF.groupby("plan")["spend"].median()


def _describe(draw):
    plt.close("all")
    draw()
    fig = plt.gcf()
    fig.savefig(io.BytesIO())
    text = describe_figure(fig)
    plt.close("all")
    return text


def _f(v):
    return f"{v:.4g}"


@pytest.mark.parametrize("orient", ["v", "h"])
def test_violins_report_each_category_from_the_inner_box(orient):
    xy = dict(x="plan", y="spend") if orient == "v" else dict(x="spend", y="plan")
    text = _describe(lambda: sns.violinplot(data=DF, **xy))
    assert "violin plot, 3 violins" in text
    for plan, median in MEDIANS.items():
        assert f"{plan}: median {_f(median)}" in text
    assert "box plot" not in text   # the inner box isn't mistaken for a box plot


def test_violin_quartile_lines_give_median_and_middle_half():
    text = _describe(lambda: sns.violinplot(data=DF, x="plan", y="spend", inner="quart"))
    assert f"basic: median {_f(MEDIANS['basic'])}, middle 50%" in text
    assert "box plot" not in text


def test_violins_split_by_hue_are_named_by_category_and_hue():
    text = _describe(lambda: sns.violinplot(data=DF, x="plan", y="spend", hue="seg"))
    assert "6 violins" in text
    assert "basic / a: median" in text and "team / b: median" in text


def test_matplotlib_violinplot_is_recognised():
    groups = [DF.spend[DF.plan == p] for p in MEDIANS.index]
    text = _describe(lambda: plt.violinplot(groups))
    assert "violin plot, 3 violins" in text


def test_lone_violin_has_no_placeholder_name():
    text = _describe(lambda: sns.violinplot(y=DF.spend))
    assert "violin plot, 1 violin: median" in text


@pytest.mark.parametrize("kind", [sns.stripplot, sns.swarmplot])
def test_strip_and_swarm_points_are_grouped_by_category(kind):
    text = _describe(lambda: kind(data=DF, x="plan", y="spend"))
    assert "points by category, 3 groups" in text
    for plan, median in MEDIANS.items():
        assert f"{plan}: 40 points" in text and f"median {_f(median)}" in text
    assert "scatter:" not in text   # no jittered x ranges like "x -0.09–0.08"


def test_symmetric_area_chart_is_not_a_violin():
    x = np.arange(11)
    text = _describe(lambda: plt.fill_between(x, 0, 25 - (x - 5) ** 2))
    assert "violin" not in text


def test_numeric_scatter_is_still_a_scatter():
    text = _describe(lambda: plt.scatter(np.arange(20), np.arange(20) ** 2))
    assert text.startswith("[chart] scatter: 20 points")


def test_executor_round_trip_describes_violins():
    code = ("df = pd.DataFrame({'g': ['x'] * 30 + ['y'] * 30, 'v': list(range(30)) + list(range(100, 130))})\n"
            "sns.violinplot(data=df, x='g', y='v')\n")
    payload = json.loads(_tool_result_for_llm(executor.run_code(code)))
    assert "violin plot, 2 violins: x: median 14.5" in payload["chart_data"][0]
