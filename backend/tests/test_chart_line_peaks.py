# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Line charts tell the agent *where* the peak and low are, with dates as dates."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from services.chartsummary import describe_figure

SALES = pd.Series([10, 30, 80, 20, 15, 12],
                  index=pd.date_range("2024-01-01", periods=6, freq="MS"), name="sales")


def _describe(draw):
    plt.close("all")
    draw()
    fig = plt.gcf()
    fig.savefig(io.BytesIO())
    text = describe_figure(fig)
    plt.close("all")
    return text


def test_pandas_monthly_series_names_peak_month():
    text = _describe(lambda: SALES.plot(title="Monthly sales"))
    assert "x 2024-01 to 2024-06" in text
    assert "max 80 at 2024-03" in text
    assert "min 10 at 2024-01" in text


def test_matplotlib_datetime_x():
    text = _describe(lambda: plt.plot(SALES.index, SALES.values))
    assert "max 80 at 2024-03-01" in text
    assert "x 2024-01-01 to 2024-06-01" in text


def test_seaborn_lineplot_dates():
    df = pd.DataFrame({"month": SALES.index, "sales": SALES.values})
    text = _describe(lambda: sns.lineplot(data=df, x="month", y="sales"))
    assert "max 80 at 2024-03-01" in text


def test_irregular_timestamps_and_times():
    s = pd.Series([1, 9, 3], index=pd.to_datetime(["2024-01-03 00:00", "2024-01-03 06:30", "2024-01-04 00:00"]))
    text = _describe(lambda: s.plot())
    assert "max 9 at 2024-01-03 06:30" in text


def test_quarterly_periods():
    text = _describe(lambda: SALES.resample("QE").sum().plot())
    assert "max 120 at 2024Q1" in text


def test_category_and_numeric_x():
    assert "max 5 at b" in _describe(lambda: plt.plot(["a", "b", "c"], [1, 5, 2]))
    text = _describe(lambda: plt.plot([2019, 2020, 2021, 2022], [5, 9, 3, 4]))
    assert "x 2019 to 2022" in text
    assert "min 3 at 2021, max 9 at 2020" in text


def test_missing_values_keep_x_aligned():
    s = pd.Series([np.nan, 4, 7, np.nan, 2], index=pd.date_range("2024-01-01", periods=5, freq="D"))
    text = _describe(lambda: plt.plot(s.index, s.values))
    assert "3 points" in text
    assert "max 7 at 2024-01-03" in text
    assert "min 2 at 2024-01-05" in text


def test_scatter_over_dates_shows_dates():
    text = _describe(lambda: plt.scatter(SALES.index, SALES.values))
    assert "x 2024-01-01 to 2024-06-01" in text
