# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Matplotlib style set by one sandbox run must not leak into later runs."""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest

import services.executor as executor


@pytest.fixture(autouse=True)
def default_style():
    # Other tests call train_model directly (outside the sandbox), which sets
    # dark_background for the whole process; start these from the defaults.
    with matplotlib.rc_context():
        plt.style.use("default")
        yield


def _corner_pixel(chart_b64: str):
    img = mpimg.imread(io.BytesIO(base64.b64decode(chart_b64)), format="png")
    return tuple(round(float(v), 2) for v in img[2, 2, :3])


def test_style_and_rcparams_do_not_leak_between_runs():
    before = (tuple(plt.rcParams["figure.figsize"]), plt.rcParams["axes.facecolor"])
    out = executor.run_code(
        "plt.rcParams['figure.figsize'] = (30, 20)\n"
        "plt.style.use('dark_background')\n"
        "plt.plot([1, 2, 3])"
    )
    assert out["error"] is None
    assert (tuple(plt.rcParams["figure.figsize"]), plt.rcParams["axes.facecolor"]) == before


def test_style_still_applies_to_the_run_that_set_it():
    dark = executor.run_code("plt.style.use('dark_background')\nplt.plot([1, 2, 3])")
    plain = executor.run_code("plt.plot([1, 2, 3])")
    assert _corner_pixel(dark["chart"]) == (0.0, 0.0, 0.0)
    assert _corner_pixel(plain["chart"]) == (1.0, 1.0, 1.0)


def test_style_is_restored_after_an_error():
    before = plt.rcParams["axes.facecolor"]
    out = executor.run_code("plt.style.use('dark_background')\n1 / 0")
    assert out["error"] and "ZeroDivisionError" in out["error"]
    assert plt.rcParams["axes.facecolor"] == before
