"""
DataForge — Auto Cleaner
Copyright (C) 2026 Mohammed Farazuddin

Automatically cleans a DataFrame and returns:
  - cleaned DataFrame
  - human-readable cleaning report (list of steps taken)
"""

import pandas as pd
import numpy as np
from typing import Any

NULL_DROP_THRESHOLD = 0.70   # drop columns with >70% missing values


def text_columns(df: pd.DataFrame, include_category: bool = False) -> list:
    """
    Text columns under both pandas 2 (dtype `object`) and pandas 3 (dtype `str`).
    `include="object"` alone misses pandas 3 string columns.
    """
    kinds = ["object", "string"] + (["category"] if include_category else [])
    return df.select_dtypes(include=kinds).columns.tolist()


def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Run the full auto-clean pipeline.
    Returns (cleaned_df, report) where report is a list of step dicts.
    """
    report = []
    df = df.copy()

    original_shape = df.shape

    # 1. Drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        report.append({
            "step": "Dropped fully empty columns",
            "detail": f"Removed {len(empty_cols)} column(s): {', '.join(empty_cols)}",
            "impact": len(empty_cols),
        })

    # 2. Drop columns with > threshold % missing
    null_ratios = df.isna().mean()
    high_null_cols = null_ratios[null_ratios > NULL_DROP_THRESHOLD].index.tolist()
    if high_null_cols:
        df.drop(columns=high_null_cols, inplace=True)
        report.append({
            "step": f"Dropped high-null columns (>{int(NULL_DROP_THRESHOLD*100)}% missing)",
            "detail": f"Removed {len(high_null_cols)} column(s): {', '.join(high_null_cols[:10])}{'…' if len(high_null_cols)>10 else ''}",
            "impact": len(high_null_cols),
        })

    # 3. Drop duplicate rows
    dupe_count = df.duplicated().sum()
    if dupe_count > 0:
        df.drop_duplicates(inplace=True)
        report.append({
            "step": "Removed duplicate rows",
            "detail": f"Dropped {dupe_count:,} duplicate row(s)",
            "impact": int(dupe_count),
        })

    # 4. Fill numeric nulls with median
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    filled_numeric = []
    for col in numeric_cols:
        null_cnt = df[col].isna().sum()
        if null_cnt > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            filled_numeric.append(f"{col} ({null_cnt:,} → median {round(median_val, 4)})")
    if filled_numeric:
        report.append({
            "step": "Filled numeric nulls with column median",
            "detail": "; ".join(filled_numeric[:10]) + ("…" if len(filled_numeric) > 10 else ""),
            "impact": len(filled_numeric),
        })

    # 5. Fill categorical nulls with mode
    cat_cols = text_columns(df, include_category=True)
    filled_cat = []
    for col in cat_cols:
        null_cnt = df[col].isna().sum()
        if null_cnt > 0:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals[0])
                filled_cat.append(f"{col} ({null_cnt:,} → '{mode_vals[0]}')")
    if filled_cat:
        report.append({
            "step": "Filled categorical nulls with column mode",
            "detail": "; ".join(filled_cat[:10]) + ("…" if len(filled_cat) > 10 else ""),
            "impact": len(filled_cat),
        })

    # 6. Strip whitespace from string columns (leaving missing values and
    # non-string cells alone — astype(str) would turn NaN into the text "nan")
    for col in text_columns(df):
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    # 7. Infer better dtypes (e.g. string numbers → int/float)
    converted = []
    for col in text_columns(df):
        try:
            df[col] = pd.to_numeric(df[col])
            converted.append(col)
        except (ValueError, TypeError):
            pass
    if converted:
        report.append({
            "step": "Converted string columns to numeric",
            "detail": f"Converted: {', '.join(converted[:10])}",
            "impact": len(converted),
        })

    final_shape = df.shape
    report.insert(0, {
        "step": "Summary",
        "detail": (
            f"Started: {original_shape[0]:,} rows × {original_shape[1]} cols  →  "
            f"Cleaned: {final_shape[0]:,} rows × {final_shape[1]} cols"
        ),
        "impact": 0,
    })

    return df, report
