"""
DataForge — Auto Plot Generator
Copyright (C) 2026 Mohammed Farazuddin

Generates a standard set of plots for any DataFrame:
  1. Missing values bar chart (before cleaning)
  2. Histograms for numeric columns  (up to 12)
  3. Correlation heatmap              (numeric, up to 20 cols)
  4. Top-value bar charts             (categorical, up to 6 cols)
  5. Box plots for numeric columns    (up to 12)

Each plot is returned as a base64-encoded PNG string.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from services.cleaner import text_columns

DARK   = "#0f1117"
SURF   = "#1a1d27"
SURF2  = "#21253a"
ACCENT = "#7c6fff"
ACCENT2= "#00d4aa"
TEXT   = "#e4e6f0"
MUTED  = "#6b7094"
DANGER = "#ff6b6b"

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=DARK)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded

def _style(fig, axes=None):
    fig.patch.set_facecolor(DARK)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(SURF2)


def plot_missing(df_original: pd.DataFrame) -> dict | None:
    """Bar chart of missing value % per column (only columns with nulls)."""
    null_pct = df_original.isna().mean() * 100
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
    if null_pct.empty:
        return None

    top = null_pct.head(30)
    fig, ax = plt.subplots(figsize=(max(8, len(top) * 0.4), 5))
    _style(fig, ax)
    colors = [DANGER if v > 70 else ACCENT for v in top.values]
    ax.bar(top.index, top.values, color=colors, width=0.6)
    ax.set_title("Missing Values by Column (%)", fontsize=12, pad=12)
    ax.set_ylabel("Missing %")
    ax.set_ylim(0, 105)
    ax.axhline(70, color=DANGER, linestyle="--", linewidth=0.8, alpha=0.6, label="70% threshold")
    ax.legend(fontsize=8, facecolor=SURF, labelcolor=TEXT)
    plt.xticks(rotation=45, ha="right")

    return {"title": "Missing Values", "formula": "Missing % = (null count / total rows) × 100", "chart": _fig_to_b64(fig)}


def plot_histograms(df: pd.DataFrame) -> dict | None:
    """Grid of histograms for numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()[:12]
    if not num_cols:
        return None

    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    _style(fig)
    axes_flat = np.array(axes).flatten() if len(num_cols) > 1 else [axes]

    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        _style(fig, ax)
        data = df[col].dropna()
        ax.hist(data, bins=30, color=ACCENT, alpha=0.85, edgecolor=SURF)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Numeric Column Distributions", fontsize=13, color=TEXT, y=1.01)
    plt.tight_layout()

    return {
        "title": "Distributions",
        "formula": "Histogram bins data into equal-width intervals. Height = frequency count per bin.",
        "chart": _fig_to_b64(fig),
    }


def plot_correlation(df: pd.DataFrame) -> dict | None:
    """Correlation heatmap for numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()[:20]
    if len(num_cols) < 2:
        return None

    corr = df[num_cols].corr()
    size = max(8, len(num_cols) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    _style(fig, ax)

    import matplotlib.colors as mcolors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "df_cmap", [DANGER, SURF2, ACCENT2], N=256
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(num_cols, fontsize=7)
    ax.set_title("Pearson Correlation Heatmap", fontsize=12, pad=12)

    # Annotate cells if small enough
    if len(num_cols) <= 12:
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color=TEXT if abs(corr.iloc[i, j]) < 0.7 else DARK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)

    return {
        "title": "Correlation Heatmap",
        "formula": "Pearson r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / (n·σₓ·σᵧ)  |  r ∈ [−1, 1]",
        "chart": _fig_to_b64(fig),
    }


def plot_categoricals(df: pd.DataFrame) -> list[dict]:
    """Bar charts for top categorical columns."""
    cat_cols = text_columns(df, include_category=True)[:6]
    results = []
    for col in cat_cols:
        vc = df[col].value_counts().head(10)
        if len(vc) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.5))
        _style(fig, ax)
        ax.barh(vc.index[::-1].astype(str), vc.values[::-1], color=ACCENT2, alpha=0.85)
        ax.set_title(f"{col} — Top Values", fontsize=10)
        ax.set_xlabel("Count")
        plt.tight_layout()
        results.append({
            "title": f"{col} Distribution",
            "formula": "Frequency count per category value.",
            "chart": _fig_to_b64(fig),
        })
    return results


def plot_boxplots(df: pd.DataFrame) -> dict | None:
    """Box plots for numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()[:12]
    if not num_cols:
        return None

    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    _style(fig)
    axes_flat = np.array(axes).flatten() if len(num_cols) > 1 else [axes]

    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        _style(fig, ax)
        data = df[col].dropna()
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": ACCENT2, "linewidth": 2},
                        boxprops={"facecolor": ACCENT, "alpha": 0.6, "linewidth": 0},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                        flierprops={"marker": "o", "color": DANGER, "markersize": 3, "alpha": 0.5})
        ax.set_title(col, fontsize=9)
        ax.set_xticks([])

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Box Plots — Spread & Outliers", fontsize=13, color=TEXT, y=1.01)
    plt.tight_layout()

    return {
        "title": "Box Plots",
        "formula": "Box = Q1→Q3 (IQR). Whiskers = Q1−1.5×IQR to Q3+1.5×IQR. Points beyond = outliers.",
        "chart": _fig_to_b64(fig),
    }


def generate_all_plots(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> list[dict]:
    """Run all plot generators and return list of {title, formula, chart}."""
    plots = []

    p = plot_missing(df_original)
    if p: plots.append(p)

    p = plot_histograms(df_cleaned)
    if p: plots.append(p)

    p = plot_correlation(df_cleaned)
    if p: plots.append(p)

    p = plot_boxplots(df_cleaned)
    if p: plots.append(p)

    plots.extend(plot_categoricals(df_cleaned))

    return plots
