# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

_ID_HINTS = re.compile(r"(^|_)(id|uuid|key|index)$", re.IGNORECASE)
_DATE_HINTS = re.compile(r"date|time|month|year|day|_at$|timestamp", re.IGNORECASE)

def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a structured EDA profile for a DataFrame.
    """
    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": [],
        "duplicates": int(df.duplicated().sum()),
        "sample": df.head(5).fillna("").to_dict(orient="records"),
        "date_column": find_date_column(df),
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "is_id": is_id_column(series, col),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info.update({
                "kind": "numeric",
                "min": _safe(desc.get("min")),
                "max": _safe(desc.get("max")),
                "mean": _safe(desc.get("mean")),
                "std": _safe(desc.get("std")),
                "median": _safe(series.median()),
                "q25": _safe(desc.get("25%")),
                "q75": _safe(desc.get("75%")),
                "outlier_count": int(_count_outliers(series)),
            })
        elif _is_date_column(series, col):
            # Dates aren't categories: "top values of signup_date" is a
            # meaningless chart, and the profile counted them as categorical
            dates = _as_dates(series).dropna()
            col_info.update({
                "kind": "datetime",
                "unique_count": int(series.nunique()),
                "min": dates.min().isoformat() if len(dates) else None,
                "max": dates.max().isoformat() if len(dates) else None,
            })
        else:
            col_info.update({
                "kind": "categorical",
                "unique_count": int(series.nunique()),
                "top_values": series.value_counts().head(5).to_dict(),
            })

        profile["columns"].append(col_info)

    return profile


def is_id_column(s: pd.Series, name) -> bool:
    """Unique-per-row identifiers (customer_id, a 1..n counter) — meaningless to plot or model."""
    n = len(s)
    if n < 2 or s.nunique() < 0.95 * n:
        return False
    if _ID_HINTS.search(str(name)):
        return True
    # Unnamed: only a true row counter (steps of exactly 1). Merely unique and
    # increasing is just as likely a real feature in data sorted by it.
    return bool(pd.api.types.is_integer_dtype(s) and s.nunique() == n
                and (s.diff().dropna() == 1).all())


def find_date_column(df: pd.DataFrame) -> str | None:
    """First column holding dates: a datetime dtype, or text named like a date that parses as one."""
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if _is_date_column(df[c], c):
            return c
    return None


def _is_date_column(s: pd.Series, name) -> bool:
    """A datetime dtype, or text named like a date that parses as one."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # object (pandas 2) or str (pandas 3) columns named like dates
    if not ((s.dtype == object or pd.api.types.is_string_dtype(s)) and _DATE_HINTS.search(str(name))):
        return False
    sample = s.dropna().astype(str).head(50)
    return not sample.empty and _as_dates(sample).notna().mean() > 0.8


def _as_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s.astype(str).where(s.notna()), errors="coerce")



def _safe(val) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return round(float(val), 4)


def _count_outliers(series: pd.Series) -> int:
    """IQR-based outlier count."""
    clean = series.dropna()
    if len(clean) < 4:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    return int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
