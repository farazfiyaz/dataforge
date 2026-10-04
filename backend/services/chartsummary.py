# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Describe a rendered matplotlib figure in text, for the agent.

The model can't see the PNGs it draws — it used to be told only "chart
rendered", so it answered "the highest bar is the answer" without knowing
which bar that was. This reads the plotted data back out of the figure:
bar labels and heights, pie shares, line and scatter ranges, heatmap value ranges.
"""

from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh
from matplotlib.container import BarContainer
from matplotlib.patches import Wedge

MAX_BARS_LISTED = 20
MAX_BOXES_LISTED = 12
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
    error_lines = set()   # seaborn error bars, described with their bars
    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    names = _legend_names(ax, bars)
    for c, name in zip(bars, names):
        items.append(_bars(ax, c, name, error_lines))
    items.append(_pie(ax))
    violin_lines = set()   # a violin's inner box, described with the violin
    items.append(_violins(ax, violin_lines))
    items.extend(_lines(ax, skip=error_lines | violin_lines))
    grouped, strip_colls = _category_points(ax)
    items.append(grouped)
    for coll in ax.collections:
        if coll in strip_colls:
            continue
        if isinstance(coll, PathCollection) and len(coll.get_offsets()):
            pts = coll.get_offsets()
            items.append(f"scatter: {len(pts)} points, "
                         f"x {_x_text(ax.xaxis, pts[:, 0].min())} to {_x_text(ax.xaxis, pts[:, 0].max())}, "
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
    if not any(items):
        return ""
    return f"[{head or 'chart'}] " + "; ".join(i for i in items if i)


def _lines(ax, skip=()) -> list[str]:
    """
    Plain lines are described one by one. Box plots are drawn as many short
    segments plus marker-only "flier" points; listing those segment by segment
    is noise, so they're summarised per box (median, middle 50%, whiskers,
    outliers), falling back to one whisker span + outlier values.
    """
    data = [(line, _nums(line.get_xdata()), _nums(line.get_ydata()))
            for line in ax.get_lines() if line not in skip]
    points = [d for d in data if _markers_only(d[0]) and d[1]]
    segments = [d for d in data if not _markers_only(d[0]) and 2 <= len(d[1]) <= 5]
    out = []
    boxes = _boxes(ax, segments, points) if len(segments) >= 4 else None
    if boxes:
        out.append(boxes)
    elif len(segments) >= 4:   # box-plot structure: box outline, whiskers, caps, median
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
        out.append(f"line{name}: {len(ys)} points" + _line_extremes(ax, line, ys))
    return out


def _boxes(ax, segments, points) -> str:
    """
    Per-box summary: median, middle 50% and whiskers of each box, by category.
    matplotlib, pandas and seaborn all draw a box as two whiskers along the
    value axis at the box's position (box edge → whisker end), plus caps and a
    median across it; the median is the widest of those. Returns "" if the
    lines don't have that structure.
    """
    best = None
    for vertical in (True, False):
        pos_of = (lambda xs, ys: xs) if vertical else (lambda xs, ys: ys)
        val_of = (lambda xs, ys: ys) if vertical else (lambda xs, ys: xs)
        whiskers, across = {}, []
        for _, xs, ys in segments:
            if len(xs) != 2 or len(ys) != 2:
                continue
            p, v = pos_of(xs, ys), val_of(xs, ys)
            if p[0] == p[1]:
                whiskers.setdefault(round(p[0], 6), []).append(v)
            elif v[0] == v[1]:
                across.append(((p[0] + p[1]) / 2, abs(p[1] - p[0]), v[0]))
        pairs = {pos: w for pos, w in whiskers.items() if len(w) == 2}
        if pairs and (best is None or len(pairs) > len(best[1])):
            best = (vertical, pairs, across)
    if best is None:
        return ""
    vertical, pairs, across = best
    labels = {round(t.get_position()[0 if vertical else 1], 6): t.get_text()
              for t in (ax.get_xticklabels() if vertical else ax.get_yticklabels()) if t.get_text()}
    names = _box_names(ax, sorted(pairs), labels)
    rows = []
    for pos, name in zip(sorted(pairs), names):
        low, high = sorted(pairs[pos], key=min)
        lo, q1, q3, hi = min(low), max(low), min(high), max(high)
        centred = [(width, v) for c, width, v in across if abs(c - pos) < 1e-6 and q1 <= v <= q3]
        if not centred:
            return ""
        median = max(centred)[1]
        fliers = sorted({(y if vertical else x)[i] for _, x, y in points
                         for i in range(min(len(x), len(y))) if abs((x if vertical else y)[i] - pos) < 1e-6})
        rows.append((name, median, q1, q3, lo, hi, fliers))
    if len(rows) == 1:
        _, median, q1, q3, lo, hi, fliers = rows[0]
        desc = f"box plot: boxes and whiskers span {_f(lo)} to {_f(hi)}; median {_f(median)}, middle 50% {_f(q1)}–{_f(q3)}"
        if fliers:
            desc += f"; outlier points at {_list(fliers, 15)}"
        return desc
    parts = []
    for label, median, q1, q3, lo, hi, fliers in rows[:MAX_BOXES_LISTED]:
        part = f"{label}: median {_f(median)}, middle 50% {_f(q1)}–{_f(q3)}, whiskers {_f(lo)}–{_f(hi)}"
        if fliers:
            part += f", outliers {_list(fliers, 5)}"
        parts.append(part)
    more = f"; +{len(rows) - MAX_BOXES_LISTED} more boxes" if len(rows) > MAX_BOXES_LISTED else ""
    return f"box plot, {len(rows)} boxes: " + "; ".join(parts) + more


def _box_names(ax, positions, labels) -> list[str]:
    """
    Tick label under each box. With seaborn `hue`, boxes are dodged off the
    ticks, `hue` values per tick in legend order, so they're named "tick / hue".
    """
    if all(p in labels for p in positions):
        return [labels[p] for p in positions]
    legend = ax.get_legend()
    hues = [t.get_text() for t in legend.get_texts()] if legend else []
    ticks = [labels[p] for p in sorted(labels)]
    if hues and ticks and len(positions) == len(hues) * len(ticks):
        return [f"{ticks[i // len(hues)]} / {hues[i % len(hues)]}" for i in range(len(positions))]
    return [labels.get(p) or _f(p) for p in positions]


def _list(values, limit) -> str:
    return ", ".join(_f(v) for v in values[:limit]) + (" …" if len(values) > limit else "")


def _line_extremes(ax, line, ys) -> str:
    """
    Range plus *where* the low and peak are, so "which month peaked?" can be
    answered from a trend chart. x is shown as the axis shows it: dates,
    periods (pandas time series), category labels or numbers.
    """
    plain = f", y from {_f(ys[0])} to {_f(ys[-1])} (min {_f(min(ys))}, max {_f(max(ys))})"
    xs, raw_ys = line.get_xdata(), line.get_ydata()
    if len(xs) != len(raw_ys):
        return plain
    kept = [i for i, y in enumerate(raw_ys) if _is_num(y)]   # same points as `ys`
    if len(kept) != len(ys):
        return plain
    low = kept[min(range(len(ys)), key=ys.__getitem__)]
    high = kept[max(range(len(ys)), key=ys.__getitem__)]
    first, last, at_low, at_high = (_x_text(ax.xaxis, xs[i]) for i in (kept[0], kept[-1], low, high))
    if None in (first, last, at_low, at_high):
        return plain
    return (f", x {first} to {last}, y from {_f(ys[0])} to {_f(ys[-1])} "
            f"(min {_f(min(ys))} at {at_low}, max {_f(max(ys))} at {at_high})")


def _x_text(axis, v):
    """One x value as the axis shows it; None if it can't be shown (e.g. a missing date)."""
    if isinstance(v, str):
        return v
    if isinstance(v, pd.Period):
        return str(v)
    if isinstance(v, (datetime, np.datetime64)):
        return None if pd.isna(v) else _date(pd.Timestamp(v))
    if not _is_num(v):
        return None
    if isinstance(axis.get_major_locator(), mdates.DateLocator):   # dates stored as day numbers
        return _date(pd.Timestamp(mdates.num2date(float(v))).tz_localize(None))
    return _f(v)


def _date(ts) -> str:
    return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.strftime("%Y-%m-%d %H:%M")


def _violins(ax, used) -> str:
    """
    Per-violin summary. A violin body is a polygon mirrored around its
    category position; seaborn's default inner box is drawn as lines at that
    position: a thin whisker, a thick middle 50%, and a one-point median;
    inner="quart" draws three lines across it instead. Those lines are added
    to `used` so they aren't read as a box plot.
    """
    found = []
    for coll in ax.collections:
        if not isinstance(coll, PolyCollection) or not coll.get_paths():
            continue
        verts = np.concatenate([p.vertices for p in coll.get_paths()])
        shape = _mirrored(verts)
        if shape:
            found.append(shape)
    if not found:
        return ""
    vertical = found[0][0]
    labels = _tick_texts(ax, vertical)
    positions = [shape[1] for shape in found]
    if labels:
        names = _box_names(ax, positions, labels)
    else:   # a lone unlabelled violin (seaborn with only y) needs no name
        names = [""] if len(found) == 1 else [_f(p) for p in positions]
    parts = []
    for (_, pos, half, lo, hi, densest), name in zip(found, names):
        inner, across = {}, []
        for line in ax.get_lines():
            p = _nums(line.get_xdata() if vertical else line.get_ydata())
            v = _nums(line.get_ydata() if vertical else line.get_xdata())
            if p and all(abs(x - pos) < 1e-6 for x in p) and len(v) in (1, 2):
                used.add(line)
                inner.setdefault(len(v), []).append((line.get_linewidth(), v))
            elif len(p) == 2 and len(v) == 2 and v[0] == v[1] and all(abs(x - pos) <= half + 1e-6 for x in p):
                used.add(line)
                across.append(v[0])
        part = f"{name}: " if name else ""
        twos = sorted(inner.get(2, []), key=lambda t: t[0])
        if inner.get(1) and len(twos) == 2:
            (_, whisk), (_, box) = twos
            part += (f"median {_f(inner[1][0][1][0])}, middle 50% {_f(min(box))}–{_f(max(box))}, "
                     f"whiskers {_f(min(whisk))}–{_f(max(whisk))}, ")
        elif len(across) == 3:
            q1, median, q3 = sorted(across)
            part += f"median {_f(median)}, middle 50% {_f(q1)}–{_f(q3)}, density from {_f(lo)} to {_f(hi)}, "
        else:
            part += f"density from {_f(lo)} to {_f(hi)}, "
        parts.append(part + f"densest near {_f(densest)}")
    more = f"; +{len(parts) - MAX_BOXES_LISTED} more violins" if len(parts) > MAX_BOXES_LISTED else ""
    count = "1 violin" if len(parts) == 1 else f"{len(parts)} violins"
    return f"violin plot, {count}: " + "; ".join(parts[:MAX_BOXES_LISTED]) + more


def _mirrored(verts):
    """
    (vertical, position, half width, low, high, densest value) if the polygon
    is a violin body, else None. Besides being mirrored, a violin narrows
    towards both ends; a symmetric area chart is mirrored too, but its
    baseline is full width.
    """
    if len(verts) < 10:
        return None
    for vertical in (True, False):
        pos, val = (verts[:, 0], verts[:, 1]) if vertical else (verts[:, 1], verts[:, 0])
        width, span = pos.max() - pos.min(), val.max() - val.min()
        if width <= 0 or span <= 0:
            continue
        centre = (pos.max() + pos.min()) / 2
        d = pos - centre
        tol = width * 1e-3
        right = {(round(v / span, 4), round(x / width, 3)) for v, x in zip(val, d) if x > tol}
        left = {(round(v / span, 4), round(-x / width, 3)) for v, x in zip(val, d) if x < -tol}
        if len(right) < 5 or len(right & left) < 0.9 * max(len(right), len(left)):
            continue
        half = np.abs(d).max()
        ends = np.isclose(val, val.min()) | np.isclose(val, val.max())
        if np.abs(d[ends]).max() >= 0.9 * half:
            continue
        return vertical, centre, half, val.min(), val.max(), val[np.argmax(np.abs(d))]
    return None


def _tick_texts(ax, vertical) -> dict:
    ticks = ax.get_xticklabels() if vertical else ax.get_yticklabels()
    return {round(t.get_position()[0 if vertical else 1], 6): t.get_text() for t in ticks if t.get_text()}


def _category_points(ax):
    """
    Strip and swarm plots: one scatter per category, jittered around the
    category's tick. Named by category instead of listing jittered x ranges.
    Returns (description, the collections it covered).
    """
    colls = [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())]
    for vertical in (True, False):
        labels = _tick_texts(ax, vertical)
        # categorical axis: tick labels that aren't just the tick positions
        if not labels or all(_is_num(_num_text(t)) and abs(float(_num_text(t)) - p) < 1e-9
                             for p, t in labels.items()):
            continue
        groups, covered = {}, []
        for c in colls:
            pts = np.asarray(c.get_offsets(), dtype=float)
            pos, val = (pts[:, 0], pts[:, 1]) if vertical else (pts[:, 1], pts[:, 0])
            near = [p for p in labels if np.all(np.abs(pos - p) < 0.5)]
            if len(near) != 1:
                continue
            groups.setdefault(near[0], []).extend(val[np.isfinite(val)])
            covered.append(c)
        if not groups:
            continue
        parts = [f"{labels[p]}: {len(v)} points, {_f(min(v))}–{_f(max(v))}, median {_f(float(np.median(v)))}"
                 for p, v in sorted(groups.items()) if v]
        more = f"; +{len(parts) - MAX_BOXES_LISTED} more groups" if len(parts) > MAX_BOXES_LISTED else ""
        return f"points by category, {len(parts)} groups: " + "; ".join(parts[:MAX_BOXES_LISTED]) + more, covered
    return "", []


def _num_text(t):
    try:
        return float(t.replace("−", "-"))
    except ValueError:
        return None


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


def _legend_names(ax, bars) -> list[str]:
    """
    Name of each bar series. pandas labels its containers; seaborn (`hue=`)
    leaves them as `_container0`, so fall back to the legend, which lists the
    series in drawing order.
    """
    names = [c.get_label() or "" for c in bars]
    names = ["" if n.startswith("_") else n for n in names]
    legend = ax.get_legend()
    if len(bars) > 1 and not any(names) and legend:
        texts = [t.get_text() for t in legend.get_texts()]
        if len(texts) == len(bars):
            return texts
    return names


def _bars(ax, container, name="", error_lines=None) -> str:
    horizontal = getattr(container, "orientation", "vertical") == "horizontal"
    patches = container.patches
    values = [p.get_width() if horizontal else p.get_height() for p in patches]
    if len(values) > MAX_BARS_LISTED:     # a histogram, most likely
        tallest = max(range(len(values)), key=values.__getitem__)
        p = patches[tallest]
        at = p.get_y() + p.get_height() / 2 if horizontal else p.get_x() + p.get_width() / 2
        return f"{len(values)} bars (histogram-like), tallest {_f(values[tallest])} at {_f(at)}"
    labels = _tick_labels(ax, patches, horizontal)
    name = f" '{name}'" if name else ""
    pairs = ", ".join(f"{lab}={_f(v)}" for lab, v in zip(labels, values))
    errors = _error_bars(ax, patches, labels, horizontal, error_lines)
    return f"bars{name}: {pairs}" + (f"; error bars: {errors}" if errors else "")


def _error_bars(ax, patches, labels, horizontal, used) -> str:
    """
    seaborn's barplot draws its confidence interval as one 2-point line
    through the centre of each bar. On their own those read as a box plot's
    whiskers (4+ bars) or as unnamed lines, so name them by bar instead.
    """
    centres = {round(p.get_y() + p.get_height() / 2 if horizontal else p.get_x() + p.get_width() / 2, 6): lab
               for p, lab in zip(patches, labels)}
    found = []
    for line in ax.get_lines():
        if used is None or line in used or _markers_only(line):
            continue
        xs, ys = _nums(line.get_xdata()), _nums(line.get_ydata())
        pos, val = (ys, xs) if horizontal else (xs, ys)
        if len(pos) == 2 and len(val) == 2 and pos[0] == pos[1] and round(pos[0], 6) in centres:
            used.add(line)
            found.append((pos[0], centres[round(pos[0], 6)], min(val), max(val)))
    return ", ".join(f"{lab} {_f(lo)}–{_f(hi)}" for _, lab, lo, hi in sorted(found))


def _tick_labels(ax, patches, horizontal) -> list[str]:
    """
    Category label under each bar (matched by position), else the bar's position.
    Grouped bars (`unstack().plot.bar()`, seaborn `hue=`) sit a fixed offset
    beside their tick, so a series whose bars are all off-tick by the same
    amount, less than half the tick spacing, takes the nearest tick's label.
    """
    ticks = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
    by_pos = {round(t.get_position()[1 if horizontal else 0], 6): t.get_text() for t in ticks if t.get_text()}
    centres = [p.get_y() + p.get_height() / 2 if horizontal else p.get_x() + p.get_width() / 2 for p in patches]
    labels = [by_pos.get(round(c, 6)) for c in centres]
    positions = sorted(by_pos)
    if not all(labels) and len(positions) >= 2:
        spacing = min(b - a for a, b in zip(positions, positions[1:]))
        nearest = [min(positions, key=lambda t: abs(t - c)) for c in centres]
        offsets = [c - t for c, t in zip(centres, nearest)]
        if (len(set(nearest)) == len(nearest) and max(offsets) - min(offsets) < 1e-6
                and abs(offsets[0]) < spacing / 2):
            labels = [by_pos[t] for t in nearest]
    return [lab or _f(c) for lab, c in zip(labels, centres)]


def _pie(ax) -> str:
    """Pie charts: each wedge's share of the whole (the raw values aren't kept by matplotlib)."""
    wedges = [p for p in ax.patches if isinstance(p, Wedge)]
    if not wedges:
        return ""
    shares = [(w.theta2 - w.theta1) / 360 * 100 for w in wedges]
    names = [w.get_label() if w.get_label() and not w.get_label().startswith("_") else f"slice {i + 1}"
             for i, w in enumerate(wedges)]
    shown = ", ".join(f"{n} {s:.1f}%" for n, s in list(zip(names, shares))[:MAX_BARS_LISTED])
    more = f", +{len(wedges) - MAX_BARS_LISTED} more" if len(wedges) > MAX_BARS_LISTED else ""
    return f"pie, {len(wedges)} slices: {shown}{more}"


def _is_num(v) -> bool:
    try:
        float(v)
        return v == v   # not NaN
    except (TypeError, ValueError):
        return False


def _f(v) -> str:
    v = float(v)
    return f"{v:.4g}"
