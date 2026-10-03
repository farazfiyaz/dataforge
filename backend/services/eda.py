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
        "sample": table_records(df, 5),
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

        # bool counts as numeric to pandas, but quantiles of True/False raise
        # in numpy, and a yes/no flag reads better as value counts anyway
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
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



def table_records(df: pd.DataFrame, n: int) -> list[dict]:
    """
    First n rows as JSON-ready records. Infinities (a spreadsheet's #DIV/0!
    saved as "inf", or a ratio column divided by zero) are not valid JSON:
    the API answered 500 and the browser couldn't parse the agent's stream.
    They're shown as the text "inf" / "-inf" instead.
    """
    head = df.head(n).copy()
    for i in range(head.shape[1]):
        s = head.iloc[:, i]
        if pd.api.types.is_float_dtype(s) and np.isinf(s).any():
            head.isetitem(i, [("inf" if v > 0 else "-inf") if isinstance(v, float) and np.isinf(v) else v
                              for v in s.astype(object)])
    return head.fillna("").to_dict(orient="records")


def finite(s: pd.Series) -> pd.Series:
    """Non-missing, finite values — matplotlib refuses to bin a range that reaches inf."""
    s = s.dropna()
    return s[np.isfinite(s.astype(float))] if pd.api.types.is_float_dtype(s) else s


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
