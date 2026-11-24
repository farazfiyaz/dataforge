# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
train_model arguments as a local model actually writes them: long model names
("random_forest"), capitalised tasks, and the target listed among features.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from services.ml import train_model


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["churn"] = np.where(df["a"] + rng.normal(size=n) > 0, "yes", "no")
    return df


@pytest.mark.parametrize("name, expected", [
    ("random_forest", "RandomForestClassifier"),
    ("RandomForest", "RandomForestClassifier"),
    ("xgboost", "GradientBoostingClassifier"),
    ("logistic", "LogisticRegression"),
    ("gb", "GradientBoostingClassifier"),
])
def test_model_aliases(name, expected):
    pipe = train_model(_df(), target="churn", model=name)
    assert type(pipe.named_steps["model"]).__name__ == expected


def test_task_is_case_insensitive():
    pipe = train_model(_df(), target="churn", task="Classification")
    assert type(pipe.named_steps["model"]).__name__ == "RandomForestClassifier"


def test_unknown_model_names_the_options():
    with pytest.raises(ValueError, match=r"unknown model 'svm'.*'gb', 'linear', 'rf'"):
        train_model(_df(), target="churn", model="svm")


def test_target_in_features_is_not_used_as_a_feature(capsys):
    pipe = train_model(_df(), target="churn", features=["a", "churn"])
    assert list(pipe.named_steps["pre"].feature_names_in_) == ["a"]
    assert "Accuracy: 1.000" not in capsys.readouterr().out


def test_only_target_as_feature_is_rejected():
    with pytest.raises(ValueError, match="other than the target"):
        train_model(_df(), target="churn", features=["churn"])


def test_missing_features_are_named():
    with pytest.raises(ValueError, match=r"features not in columns: \['zzz'\]"):
        train_model(_df(), target="churn", features=["a", "zzz"])


def test_single_feature_string():
    pipe = train_model(_df(), target="churn", features="a")
    assert list(pipe.named_steps["pre"].feature_names_in_) == ["a"]
