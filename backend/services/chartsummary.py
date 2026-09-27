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
    items.extend(_lines(ax))
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and len(coll.get_offsets()):
            pts = coll.get_offsets()
            items.append(f"scatter: {len(pts)} points, x {_f(pts[:, 0].min())}–{_f(pts[:, 0].max())}, "
                         f"y {_f(pts[:, 1].min())}–{_f(pts[:, 1].max())}")
        elif isinstance(coll, QuadMesh):
            arr = coll.get_array()
            if arr is not None and arr.size:
                items.append(f"heatmap: {arr.size} cells, values {_f(arr.min())} to {_f(arr.max())}"
                             + _strongest_cells(ax, arr))
    for img in ax.images:
        arr = img.get_array()
        if arr is not None and arr.size:
            items.append(f"image/heatmap: values {_f(arr.min())} to {_f(arr.max())}")
    if not items:
        return ""
    return f"[{head or 'chart'}] " + "; ".join(i for i in items if i)


def _lines(ax) -> list[str]:
    """
    Plain lines are described one by one. Box plots are drawn as many short
    segments plus marker-only "flier" points; listing those segment by segment
    is noise, so they're summarised as whisker span + outlier values.
    """
    data = [(line, _nums(line.get_xdata()), _nums(line.get_ydata())) for line in ax.get_lines()]
    points = [d for d in data if _markers_only(d[0]) and d[1]]
    segments = [d for d in data if not _markers_only(d[0]) and 2 <= len(d[1]) <= 5]
    out = []
    if len(segments) >= 4:   # box-plot structure: box outline, whiskers, caps, median
        xs = [v for _, x, _ in segments for v in x]
        ys = [v for _, _, y in segments for v in y]
        vertical = (max(ys) - min(ys)) >= (max(xs) - min(xs))
        span = ys if vertical else xs
        desc = f"box plot: boxes and whiskers span {_f(min(span))} to {_f(max(span))}"
        fliers = sorted({v for _, x, y in points for v in (y if vertical else x)})
        if fliers:
            shown = ", ".join(_f(v) for v in fliers[:15]) + (" …" if len(fliers) > 15 else "")
            desc += f"; outlier points at {shown}"
        out.append(desc)
    else:
        segments = []
    for line, xs, ys in data:
        if len(ys) < 2 or (line, xs, ys) in segments or (segments and _markers_only(line)):
            continue
        name = line.get_label()
        name = "" if name.startswith("_") else f" '{name}'"
        out.append(f"line{name}: {len(ys)} points, y from {_f(ys[0])} to {_f(ys[-1])} "
                   f"(min {_f(min(ys))}, max {_f(max(ys))})")
    return out


def _strongest_cells(ax, arr) -> str:
    """For a labelled square heatmap (e.g. a correlation matrix), name the strongest off-diagonal pairs."""
    xl = [t.get_text() for t in ax.get_xticklabels()]
    yl = [t.get_text() for t in ax.get_yticklabels()]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or len(xl) != arr.shape[1] or len(yl) != arr.shape[0]:
        return ""
    cells = [(abs(float(arr[i, j])), float(arr[i, j]), yl[i], xl[j])
             for i in range(arr.shape[0]) for j in range(i + 1, arr.shape[1]) if _is_num(arr[i, j])]
    top = sorted(cells, reverse=True)[:3]
    return "; strongest pairs: " + ", ".join(f"{a}–{b} {_f(v)}" for _, v, a, b in top) if top else ""


def _markers_only(line) -> bool:
    return line.get_linestyle() in ("None", "none", "", " ") and line.get_marker() not in (None, "None", "none", "")


def _nums(values) -> list[float]:
    return [float(v) for v in values if _is_num(v)]


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
