# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import pandas as pd
import numpy as np
from typing import Any

def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a structured EDA profile for a DataFrame.
    """
    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": [],
        "duplicates": int(df.duplicated().sum()),
        "sample": df.head(5).fillna("").to_dict(orient="records"),
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
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
        else:
            col_info.update({
                "kind": "categorical",
                "unique_count": int(series.nunique()),
                "top_values": series.value_counts().head(5).to_dict(),
            })

        profile["columns"].append(col_info)

    return profile


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
