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


def test_correlation_heatmap_names_strongest_pairs():
    corr = pd.DataFrame([[1, 0.9, -0.2], [0.9, 1, 0.1], [-0.2, 0.1, 1]],
                        index=list("abc"), columns=list("abc"))
    text = _describe(lambda: sns.heatmap(corr))
    assert "strongest pairs: a–b 0.9, a–c -0.2, b–c 0.1" in text


def test_lines_with_markers_are_still_lines():
    assert "line: 3 points" in _describe(lambda: plt.plot([1, 2, 3], [3, 1, 2], "o-"))
