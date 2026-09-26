# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Describe a rendered matplotlib figure in text, for the agent.

The model can't see the PNGs it draws — it used to be told only "chart
rendered", so it answered "the highest bar is the answer" without knowing
which bar that was. This reads the plotted data back out of the figure:
bar labels and heights, line and scatter ranges, heatmap value ranges.
"""

from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.container import BarContainer

MAX_BARS_LISTED = 20
MAX_CHARS = 1500


def describe_figure(fig) -> str:
    """Text summary of every axes in `fig`. Call after the figure is drawn (e.g. savefig)."""
    parts = []
    for ax in fig.get_axes():
        if ax.get_label() == "<colorbar>":
            continue
        desc = _describe_axes(ax)
        if desc:
            parts.append(desc)
    text = "\n".join(parts)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " …"


def _describe_axes(ax) -> str:
    head = ", ".join(f"{k} '{v}'" for k, v in (
        ("title", ax.get_title()), ("x", ax.get_xlabel()), ("y", ax.get_ylabel())) if v)
    items = []
    for c in ax.containers:
        if isinstance(c, BarContainer):
            items.append(_bars(ax, c))
    for line in ax.get_lines():
        ys = [float(y) for y in line.get_ydata() if _is_num(y)]
        if len(ys) > 1:
            name = line.get_label()
            name = "" if name.startswith("_") else f" '{name}'"
            items.append(f"line{name}: {len(ys)} points, y from {_f(ys[0])} to {_f(ys[-1])} "
                         f"(min {_f(min(ys))}, max {_f(max(ys))})")
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and len(coll.get_offsets()):
            pts = coll.get_offsets()
            items.append(f"scatter: {len(pts)} points, x {_f(pts[:, 0].min())}–{_f(pts[:, 0].max())}, "
                         f"y {_f(pts[:, 1].min())}–{_f(pts[:, 1].max())}")
        elif isinstance(coll, QuadMesh):
            arr = coll.get_array()
            if arr is not None and arr.size:
                items.append(f"heatmap: {arr.size} cells, values {_f(arr.min())} to {_f(arr.max())}")
    for img in ax.images:
        arr = img.get_array()
        if arr is not None and arr.size:
            items.append(f"image/heatmap: values {_f(arr.min())} to {_f(arr.max())}")
    if not items:
        return ""
    return f"[{head or 'chart'}] " + "; ".join(i for i in items if i)


def _bars(ax, container) -> str:
    horizontal = getattr(container, "orientation", "vertical") == "horizontal"
    patches = container.patches
    values = [p.get_width() if horizontal else p.get_height() for p in patches]
    if len(values) > MAX_BARS_LISTED:     # a histogram, most likely
        tallest = max(range(len(values)), key=values.__getitem__)
        p = patches[tallest]
        at = p.get_y() + p.get_height() / 2 if horizontal else p.get_x() + p.get_width() / 2
        return f"{len(values)} bars (histogram-like), tallest {_f(values[tallest])} at {_f(at)}"
    labels = _tick_labels(ax, patches, horizontal)
    name = container.get_label()
    name = "" if not name or name.startswith("_") else f" '{name}'"
    pairs = ", ".join(f"{lab}={_f(v)}" for lab, v in zip(labels, values))
    return f"bars{name}: {pairs}"


def _tick_labels(ax, patches, horizontal) -> list[str]:
    """Category label under each bar (matched by position), else the bar's position."""
    ticks = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
    by_pos = {round(t.get_position()[1 if horizontal else 0], 6): t.get_text() for t in ticks if t.get_text()}
    labels = []
    for p in patches:
        centre = p.get_y() + p.get_height() / 2 if horizontal else p.get_x() + p.get_width() / 2
        labels.append(by_pos.get(round(centre, 6)) or _f(centre))
    return labels


def _is_num(v) -> bool:
    try:
        float(v)
        return v == v   # not NaN
    except (TypeError, ValueError):
        return False


def _f(v) -> str:
    v = float(v)
    return f"{v:.4g}"
