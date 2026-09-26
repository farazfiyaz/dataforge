# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
ML pipeline builder.

One call trains a full scikit-learn pipeline with sensible preprocessing,
prints a metrics report, and draws diagnostic charts. Designed to be called
by the agent inside the sandbox:

    model = train_model(df, target="churn")            # auto everything
    model = train_model(df, target="price", model="gb") # gradient boosting

The returned object is a fitted sklearn Pipeline — usable for .predict().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
)

MAX_ONEHOT_CARDINALITY = 20   # categorical columns with more uniques get dropped
MAX_FEATURE_BARS = 15         # top-N features shown in the importance chart

_MODELS = {
    ("classification", "rf"):     lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    ("classification", "gb"):     lambda: GradientBoostingClassifier(random_state=42),
    ("classification", "linear"): lambda: LogisticRegression(max_iter=2000),
    ("regression", "rf"):         lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    ("regression", "gb"):         lambda: GradientBoostingRegressor(random_state=42),
    ("regression", "linear"):     lambda: LinearRegression(),
}


def train_model(df: pd.DataFrame, target: str, task: str = None,
                model: str = "rf", features: list = None, test_size: float = 0.2):
    """
    Train a model to predict `target` from the other columns.

    task:    "classification" | "regression" | None (auto-detect)
    model:   "rf" (random forest, default) | "gb" (gradient boosting) | "linear"
    features: optional list of columns to use (default: all except target)

    Prints a metrics report, draws diagnostic charts, returns the fitted Pipeline.
    """
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns: {list(df.columns)}")

    data = df.dropna(subset=[target]).copy()
    y = data[target]

    # ── auto-detect task ──
    if task is None:
        task = "classification" if (y.dtype == object or y.dtype == bool
                                    or y.nunique() <= 10) else "regression"

    label_encoder = None
    if task == "classification" and y.dtype == object:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=data.index)

    # ── feature selection ──
    X = data[features] if features else data.drop(columns=[target])
    X = X.select_dtypes(exclude=["datetime64[ns]", "datetime64[ns, UTC]"])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.select_dtypes(include=["object", "category", "bool"]).columns
                if X[c].nunique() <= MAX_ONEHOT_CARDINALITY]
    dropped = [c for c in X.columns if c not in num_cols + cat_cols]
    X = X[num_cols + cat_cols]
    if X.empty:
        raise ValueError("no usable feature columns found")

    # ── preprocessing + model pipeline ──
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    est = _MODELS[(task, model)]()
    pipe = Pipeline([("pre", pre), ("model", est)])

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=42, stratify=stratify)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    # ── report + charts ──
    plt.style.use("dark_background")
    print(f"=== {task.upper()} — {type(est).__name__} ===")
    print(f"Rows: {len(data):,}  |  Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical"
          + (f"  (dropped: {dropped})" if dropped else ""))

    if task == "classification":
        _report_classification(pipe, X_te, y_te, y_pred, label_encoder)
    else:
        _report_regression(y_te, y_pred, pipe, num_cols, cat_cols)

    return pipe


def _feature_importances(pipe, ax):
    """Bar chart of top features by importance (tree models) or |coef| (linear)."""
    est = pipe.named_steps["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
    else:
        ax.axis("off"); return
    order = np.argsort(imp)[::-1][:MAX_FEATURE_BARS]
    ax.barh([names[i].split("__")[-1] for i in order][::-1], imp[order][::-1], color="#4f9cf9")
    ax.set_title("Feature importance")


def _report_classification(pipe, X_te, y_te, y_pred, label_encoder):
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted",
                                                       zero_division=0)
    print(f"Accuracy: {acc:.3f}  |  Precision: {prec:.3f}  |  "
          f"Recall: {rec:.3f}  |  F1: {f1:.3f}")

    binary = len(np.unique(y_te)) == 2
    auc = None
    if binary and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        print(f"ROC AUC: {auc:.3f}")

    fig, axes = plt.subplots(1, 3 if binary else 2, figsize=(15 if binary else 10, 4))
    fig.suptitle("Model diagnostics")

    # confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    labels = label_encoder.classes_ if label_encoder is not None else np.unique(y_te)
    ax = axes[0]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black")
    ax.set_title("Confusion matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ROC curve (binary only)
    if binary and auc is not None:
        fpr, tpr, _ = roc_curve(y_te, proba)
        axes[1].plot(fpr, tpr, color="#4f9cf9")
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].set_title(f"ROC curve (AUC={auc:.3f})")
        axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")

    _feature_importances(pipe, axes[-1])
    fig.tight_layout()


def _report_regression(y_te, y_pred, pipe, num_cols, cat_cols):
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    print(f"R²: {r2:.3f}  |  MAE: {mae:,.3f}  |  RMSE: {rmse:,.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Model diagnostics")

    axes[0].scatter(y_te, y_pred, s=8, alpha=0.5, color="#4f9cf9")
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    axes[0].plot(lims, lims, "--", color="gray")
    axes[0].set_title(f"Predicted vs actual (R²={r2:.3f})")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")

    residuals = y_te - y_pred
    axes[1].hist(residuals, bins=40, color="#4f9cf9")
    axes[1].set_title("Residuals"); axes[1].set_xlabel("Actual − predicted")

    _feature_importances(pipe, axes[2])
    fig.tight_layout()
