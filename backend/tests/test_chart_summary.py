# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""The agent gets a text description of each chart, since it can't see images."""
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

RATES = pd.Series({"basic": 0.5769, "enterprise": 0.4141, "pro": 0.4422})


def _describe(draw):
    plt.close("all")
    draw()
    fig = plt.gcf()
    fig.savefig(io.BytesIO())
    text = describe_figure(fig)
    plt.close("all")
    return text


@pytest.mark.parametrize("draw", [
    lambda: plt.bar(RATES.index, RATES.values),
    lambda: RATES.plot(kind="bar"),
    lambda: sns.barplot(x=RATES.index, y=RATES.values),
])
def test_bar_charts_list_each_label_and_value(draw):
    assert "basic=0.5769, enterprise=0.4141, pro=0.4422" in _describe(draw)


def test_horizontal_bars_and_titles():
    text = _describe(lambda: RATES.plot(kind="barh", title="Churn", xlabel="rate"))
    assert "title 'Churn'" in text and "basic=0.5769" in text


def test_histogram_is_summarised_not_listed():
    text = _describe(lambda: plt.hist(np.random.default_rng(0).normal(size=500), bins=40))
    assert "40 bars (histogram-like)" in text


def test_line_scatter_heatmap():
    assert "line 'y': 5 points" in _describe(lambda: plt.plot(range(5), [1, 4, 2, 8, 5], label="y"))
    assert "scatter: 3 points" in _describe(lambda: plt.scatter([1, 2, 3], [4, 5, 6]))
    text = _describe(lambda: sns.heatmap(np.array([[1.0, 0.5], [0.5, 1.0]])))
    assert "heatmap: 4 cells, values 0.5 to 1" in text
    assert text.count("heatmap") == 1          # colorbar isn't described as a second heatmap


def test_executor_passes_chart_data_to_the_model():
    result = executor.run_code("plt.bar(['a', 'b'], [3, 7])\nplt.title('T')")
    payload = json.loads(_tool_result_for_llm(result))
    assert payload["chart"] == "rendered and shown to the user"
    assert payload["chart_data"] == ["[title 'T'] bars: a=3, b=7"]


TICKETS = pd.Series([0] * 30 + [1] * 50 + [2] * 40 + [3] * 15 + [4] * 5 + [7])


@pytest.mark.parametrize("draw", [
    lambda: sns.boxplot(x=TICKETS),
    lambda: sns.boxplot(y=TICKETS),
    lambda: plt.boxplot(TICKETS),
])
def test_box_plots_report_whiskers_and_outliers(draw):
    text = _describe(draw)
    assert "box plot: boxes and whiskers span 0 to 3" in text
    assert "outlier points at 4, 7" in text
    assert "line:" not in text                   # no segment-by-segment noise


def test_single_box_reports_median_and_middle_half():
    text = _describe(lambda: sns.boxplot(y=TICKETS))
    assert "median 1, middle 50% 1–2" in text


PLANS = pd.DataFrame({
    "plan": ["basic"] * 7 + ["pro"] * 7 + ["enterprise"] * 7,
    "charges": [10, 11, 12, 13, 14, 15, 60] + [20, 22, 24, 26, 28, 30, 32] + [40, 41, 42, 43, 44, 45, 46],
})
GROUPED = ("box plot, 3 boxes: basic: median 13, middle 50% 11.5–14.5, whiskers 10–15, outliers 60; "
           "pro: median 26, middle 50% 23–29, whiskers 20–32; "
           "enterprise: median 43, middle 50% 41.5–44.5, whiskers 40–46")


@pytest.mark.parametrize("draw", [
    lambda: sns.boxplot(data=PLANS, x="plan", y="charges"),
    lambda: sns.boxplot(data=PLANS, x="charges", y="plan"),
    lambda: plt.boxplot([PLANS.charges[PLANS.plan == p] for p in ("basic", "pro", "enterprise")],
                        tick_labels=["basic", "pro", "enterprise"]),
])
def test_grouped_box_plots_are_summarised_per_box(draw):
    # One combined "5.35 to 46" span didn't say which plan was higher.
    assert GROUPED in _describe(draw)


def test_pandas_grouped_box_plot_uses_its_own_order():
    text = _describe(lambda: PLANS.boxplot(column="charges", by="plan"))
    assert "basic: median 13" in text and "enterprise: median 43" in text and "pro: median 26" in text


def test_hue_box_plots_name_category_and_hue():
    df = pd.DataFrame({"churn": ["no"] * 8 + ["yes"] * 8,
                       "plan": (["basic"] * 4 + ["pro"] * 4) * 2,
                       "tenure": [50, 52, 54, 56, 40, 42, 44, 46, 10, 12, 14, 16, 20, 22, 24, 26]})
    text = _describe(lambda: sns.boxplot(data=df, x="churn", y="tenure", hue="plan"))
    assert "box plot, 4 boxes: no / basic: median 53" in text
    assert "no / pro: median 43" in text and "yes / basic: median 13" in text and "yes / pro: median 23" in text


def test_correlation_heatmap_names_strongest_pairs():
    corr = pd.DataFrame([[1, 0.9, -0.2], [0.9, 1, 0.1], [-0.2, 0.1, 1]],
                        index=list("abc"), columns=list("abc"))
    text = _describe(lambda: sns.heatmap(corr))
    assert "strongest pairs: a–b 0.9, a–c -0.2, b–c 0.1" in text


def test_lines_with_markers_are_still_lines():
    assert "line: 3 points" in _describe(lambda: plt.plot([1, 2, 3], [3, 1, 2], "o-"))
