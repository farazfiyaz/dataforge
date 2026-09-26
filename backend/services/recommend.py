# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Next-step recommendations — "what should I do with this data?"

Rule-based on purpose: they come straight from the data (missing values,
likely target columns, strong correlations, outliers, dates), so they are
instant, deterministic, and don't depend on how capable the local LLM is.
Each recommendation carries a ready-to-send `prompt` for the agent.
"""

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

# Column names that usually mean "this is what you'd want to predict"
_TARGET_HINTS = re.compile(
    r"(^|_)(churn(ed)?|target|label|class|outcome|default(ed)?|fraud|survived|"
    r"converted|conversion|attrition|response|y)$|^(is|has)_",
    re.IGNORECASE,
)
_ID_HINTS = re.compile(r"(^|_)(id|uuid|key|index)$", re.IGNORECASE)
_DATE_HINTS = re.compile(r"date|time|month|year|day|_at$|timestamp", re.IGNORECASE)
_YES_NO = {"0", "1", "0.0", "1.0", "yes", "no", "y", "n", "true", "false", "t", "f"}

MISSING_PCT_THRESHOLD = 5.0      # flag columns with more missing values than this
OUTLIER_SHARE_THRESHOLD = 0.05   # flag numeric columns with >5% IQR outliers
CORR_THRESHOLD = 0.5             # flag numeric pairs with |r| above this
MAX_CORR_COLS = 30               # cap the correlation matrix on wide data


def recommend(df: pd.DataFrame, asked: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """
    Return up to `limit` recommendations, most useful first.

    Recommendations whose prompt the user has already sent (in `asked`) are
    skipped, so follow-up suggestions don't repeat themselves.
    """
    asked_set = {a.strip().lower() for a in (asked or [])}
    recs = [r for r in _all_recommendations(df) if r["prompt"].strip().lower() not in asked_set]
    recs.sort(key=lambda r: r["_score"], reverse=True)
    return [{k: v for k, v in r.items() if k != "_score"} for r in recs[:limit]]


def _rec(rec_id: str, title: str, reason: str, prompt: str, score: float) -> dict[str, Any]:
    return {"id": rec_id, "title": title, "reason": reason, "prompt": prompt, "_score": score}


def _all_recommendations(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or len(df.columns) == 0:
        return []

    recs: list[dict[str, Any]] = []
    rows = len(df)
    id_cols = [c for c in df.columns if _is_id_column(df[c], c)]
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in id_cols]
    target = _find_target(df, exclude=id_cols)
    categoricals = [
        c for c in df.columns
        if c not in id_cols and c != target and not pd.api.types.is_numeric_dtype(df[c])
        and 2 <= df[c].nunique() <= 12
    ]
    date_col = _find_date_column(df)
    exclude_note = f" Exclude {_names(id_cols)} — it's an identifier." if id_cols else ""

    # ── Predict the likely target ──
    if target is not None:
        counts = df[target].value_counts(normalize=True)
        reason = f"`{target}` looks like an outcome column ({len(counts)} classes)."
        if len(counts) == 2 and counts.min() < 0.2:
            reason += f" It's imbalanced: only {counts.min():.0%} are `{counts.idxmin()}`."
        recs.append(_rec(
            "predict-target", f"Predict {target}", reason,
            f"Train a model to predict `{target}` from the other columns.{exclude_note} "
            "Report the metrics in plain English and tell me which features matter most.",
            95,
        ))
        if categoricals:
            seg = categoricals[0]
            recs.append(_rec(
                "segment-target", f"Compare {target} across {seg}",
                f"See which `{seg}` groups differ most on `{target}`.",
                f"Compare the rate of `{target}` across each `{seg}` group with a bar chart, "
                "and tell me which group stands out.",
                80,
            ))

    # ── Missing values ──
    null_pct = df.isna().mean() * 100
    worst = null_pct[null_pct > MISSING_PCT_THRESHOLD].sort_values(ascending=False)
    if not worst.empty:
        cols = list(worst.index[:3])
        recs.append(_rec(
            "missing-values", f"Handle missing values in {', '.join(map(str, cols))}",
            f"`{worst.index[0]}` is {worst.iloc[0]:.0f}% missing, which can skew any analysis or model.",
            f"Look at the missing values in {_names(cols)}. Recommend whether to drop or impute each one, "
            "explain why, then apply it to `df`.",
            90 if worst.iloc[0] > 20 else 70,
        ))

    # ── Duplicates ──
    dupes = int(df.duplicated().sum())
    if dupes:
        recs.append(_rec(
            "duplicates", f"Remove {dupes:,} duplicate rows",
            f"{dupes / rows:.1%} of rows are exact duplicates.",
            "Show me a few of the duplicated rows, then remove the duplicates from `df` and tell me the new row count.",
            75,
        ))

    # ── Strongest correlation ──
    pair = _strongest_correlation(df, numeric)
    if pair is not None:
        a, b, r = pair
        recs.append(_rec(
            "correlation", f"Explore {a} vs {b}",
            f"They're strongly {'positively' if r > 0 else 'negatively'} correlated (r = {r:.2f}).",
            f"Plot `{a}` against `{b}` with a regression line and explain the relationship.",
            65 + 20 * abs(r),
        ))

    # ── Outliers ──
    outlier_col, share = _most_outliers(df, numeric)
    if outlier_col is not None:
        recs.append(_rec(
            "outliers", f"Investigate outliers in {outlier_col}",
            f"{share:.0%} of `{outlier_col}` values fall outside the IQR fences.",
            f"Show a box plot of `{outlier_col}`, list the most extreme values, and tell me whether "
            "they look like errors or genuine extremes.",
            60,
        ))

    # ── Trend over time ──
    if date_col is not None and numeric:
        metric = target if target in numeric else numeric[0]
        recs.append(_rec(
            "trend", f"Show {metric} over time",
            f"`{date_col}` looks like a date, so the data can be viewed as a time series.",
            f"Parse `{date_col}` as a date and plot how `{metric}` changes over time, aggregated sensibly.",
            72,
        ))

    # ── Always-available starting points ──
    recs.append(_rec(
        "summary", "Summarize the dataset",
        "A quick overview of every column, its type, and anything unusual.",
        "Give me a concise summary of this dataset: what each column is, notable statistics, "
        "and the three most interesting things you notice.",
        40,
    ))
    if len(numeric) >= 3:
        recs.append(_rec(
            "heatmap", "Correlation heatmap",
            f"See how all {len(numeric)} numeric columns relate at a glance.",
            "Draw a correlation heatmap of the numeric columns and point out the strongest relationships.",
            45,
        ))
    return recs


def _names(cols: list) -> str:
    quoted = [f"`{c}`" for c in cols]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _is_id_column(s: pd.Series, name) -> bool:
    n = len(s)
    if n < 2 or s.nunique() < 0.95 * n:
        return False
    if _ID_HINTS.search(str(name)):
        return True
    return bool(pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing and s.nunique() == n)


def _find_target(df: pd.DataFrame, exclude: list) -> str | None:
    candidates = [c for c in df.columns if c not in exclude and 2 <= df[c].nunique() <= 10]
    for c in candidates:              # a name that says "target" wins
        if _TARGET_HINTS.search(str(c)):
            return c
    # Otherwise only a yes/no-style column — a two-valued column like
    # `region` (N/S) is a grouping, not something you'd predict
    binary = [c for c in candidates
              if df[c].nunique() == 2 and {str(v).strip().lower() for v in df[c].dropna().unique()} <= _YES_NO]
    return binary[-1] if binary else None   # outcome columns tend to come last


def _find_date_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        # object (pandas 2) or str (pandas 3) columns named like dates
        if (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])) and _DATE_HINTS.search(str(c)):
            sample = df[c].dropna().astype(str).head(50)
            if sample.empty:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return c
    return None


def _strongest_correlation(df: pd.DataFrame, numeric: list) -> tuple[str, str, float] | None:
    cols = [c for c in numeric[:MAX_CORR_COLS] if df[c].nunique() > 2]
    if len(cols) < 2:
        return None
    corr = df[cols].corr().abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    if corr[i, j] < CORR_THRESHOLD:
        return None
    a, b = cols[i], cols[j]
    return a, b, float(df[a].corr(df[b]))


def _most_outliers(df: pd.DataFrame, numeric: list) -> tuple[str | None, float]:
    best, best_share = None, 0.0
    for c in numeric:
        s = df[c].dropna()
        if len(s) < 20 or s.nunique() <= 2:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        share = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
        if share > best_share:
            best, best_share = c, share
    return (best, best_share) if best_share > OUTLIER_SHARE_THRESHOLD else (None, 0.0)
