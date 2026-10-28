"""Static figures for results and the brief.

Palette and mark rules follow the dataviz reference instance: categorical slots in fixed
order (first three validate all-pairs for colour-vision deficiency), 2 px lines, recessive
grid, one y-axis per chart, a legend for two or more series plus direct end labels, text in
text colours (never the series colour). Every figure has a CSV table next to it.
"""

from __future__ import annotations

import csv
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (light mode)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
LINE_PT = 1.5  # 2 px at 96 dpi
MARKER_PT = 6.0  # 8 px at 96 dpi


def _pretty(name: str) -> str:
    """Display label for a series key ("oracle_plane" -> "Oracle plane")."""
    s = name.replace("_", " ")
    return s[:1].upper() + s[1:]


def _note(fig: plt.Figure, note: str | None) -> float:
    """Wrapped footnote at the bottom; returns the figure fraction it occupies."""
    if not note:
        return 0.0
    text = textwrap.fill(note, width=105)
    fig.text(0.01, 0.01, text, color=TEXT_2, fontsize=7, ha="left", va="bottom")
    return 0.035 * len(text.splitlines()) + 0.01


LABEL_GAP_PX = 12  # minimum vertical spacing between direct end labels


def _end_labels(fig: plt.Figure, ax: plt.Axes, ends: list[tuple[str, tuple[float, float]]]) -> None:
    """Direct labels at series ends, skipping any that would collide with one already placed
    (the legend still identifies every series, so nothing is lost)."""
    fig.canvas.draw()
    placed: list[float] = []
    for label, (x, y) in ends:
        py = ax.transData.transform((x, y))[1]
        if any(abs(py - q) < LABEL_GAP_PX for q in placed):
            continue
        placed.append(py)
        ax.annotate(
            label,
            (x, y),
            xytext=(4, 0),
            textcoords="offset points",
            color=TEXT_2,
            fontsize=8,
            va="center",
        )


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def line_chart(
    path: Path,
    series: Mapping[str, Mapping[str, Sequence[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
    logy: bool = False,
    note: str | None = None,
) -> None:
    """Lines with optional shaded intervals. `series[name]` has x, y and optional lo, hi.
    At most three series (validated palette cap). Writes `path` (.png) and a .csv table."""
    if len(series) > len(SERIES):
        raise ValueError("at most three series per chart; facet instead")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    rows = []
    ends: list[tuple[str, tuple[float, float]]] = []
    for (name, s), colour in zip(series.items(), SERIES, strict=False):
        x, y = list(s["x"]), list(s["y"])
        if "lo" in s and "hi" in s:
            ax.fill_between(x, s["lo"], s["hi"], color=colour, alpha=0.15, linewidth=0)
        ax.plot(
            x,
            y,
            color=colour,
            linewidth=LINE_PT,
            marker="o",
            markersize=MARKER_PT,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=_pretty(name),
        )
        finite = [(a, b) for a, b in zip(x, y, strict=True) if b == b]
        if finite and len(series) <= 4:
            ends.append((_pretty(name), finite[-1]))
        for i, (a, b) in enumerate(zip(x, y, strict=True)):
            rows.append(
                {
                    "series": name,
                    "x": a,
                    "y": b,
                    "lo": s["lo"][i] if "lo" in s else "",
                    "hi": s["hi"][i] if "hi" in s else "",
                }
            )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _style(ax, title, xlabel, ylabel)
    ax.margins(x=0.18)  # room for direct end labels
    if len(series) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    _end_labels(fig, ax, ends)
    bottom = _note(fig, note)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    write_csv(path.with_suffix(".csv"), rows)


def bar_chart(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    note: str | None = None,
) -> None:
    """Horizontal bars, largest first (tornado chart). One series, so no legend."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    fig, ax = plt.subplots(figsize=(6.4, 0.4 * len(values) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.barh(
        [labels[i] for i in order],
        [values[i] for i in order],
        color=SERIES[0],
        height=0.6,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    for k, i in enumerate(order):
        ax.annotate(
            f"{values[i]:#.3g}",
            (values[i], k),
            xytext=(4, 0),
            textcoords="offset points",
            color=TEXT_2,
            fontsize=8,
            va="center",
        )
    _style(ax, title, xlabel, "")
    bottom = _note(fig, note)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    write_csv(
        path.with_suffix(".csv"),
        [{"label": a, "value": b} for a, b in zip(labels, values, strict=True)],
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows (dicts with the same keys) as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


MUTED = "#86857f"  # dominated points: recessive grey (not a slot), >= 3:1 on SURFACE


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated by another (lower x is better, higher y is better),
    sorted by x."""
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best_y = [], float("-inf")
    for i in idx:
        if points[i][1] > best_y:
            front.append(i)
            best_y = points[i][1]
    return front


def pareto_chart(
    path: Path,
    names: Sequence[str],
    x: Sequence[float],
    y: Sequence[float],
    budgets: Mapping[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    note: str | None = None,
    logx: bool = False,
) -> None:
    """Accuracy against latency. Frontier points in series slot 1, joined and
    directly labelled (a label that would collide with one already placed is skipped; the CSV
    lists every point); dominated points muted and unlabelled. Budgets are dashed vertical
    reference lines labelled in text colour."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    front = pareto_frontier(list(zip(x, y, strict=True)))
    others = [i for i in range(len(x)) if i not in front]
    ax.scatter(
        [x[i] for i in others],
        [y[i] for i in others],
        s=MARKER_PT**2,
        color=MUTED,
        edgecolor=SURFACE,
        linewidth=1.0,
        zorder=2,
        label="Dominated",
    )
    ax.plot(
        [x[i] for i in front],
        [y[i] for i in front],
        color=SERIES[0],
        linewidth=LINE_PT,
        marker="o",
        markersize=MARKER_PT,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        zorder=3,
        label="Best trade-off",
    )
    for label, b in budgets.items():
        ax.axvline(b, color=TEXT_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            label,
            (b, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4),
            textcoords="offset points",
            color=TEXT_2,
            fontsize=7,
            va="top",
        )
    _style(ax, title, xlabel, ylabel)
    if logx:
        ax.set_xscale("log")
        lo, hi = min([*x, *budgets.values()]) * 0.8, max(x) * 1.25
        ticks = [t for t in (20, 33, 50, 100, 200, 500, 1000) if lo <= t <= hi]
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    else:
        ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="lower left")
    fig.canvas.draw()
    placed: list[tuple[float, float]] = []
    for i in front:
        px, py = ax.transData.transform((x[i], y[i]))
        if any(abs(px - qx) < 120 and abs(py - qy) < 14 for qx, qy in placed):
            continue
        placed.append((px, py))
        ax.annotate(
            names[i],
            (x[i], y[i]),
            xytext=(6, -4),
            textcoords="offset points",
            color=TEXT_2,
            fontsize=7,
            va="top",
        )
    bottom = _note(fig, note)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    write_csv(
        path.with_suffix(".csv"),
        [
            {"config": n, "x": a, "y": b, "frontier": i in front}
            for i, (n, a, b) in enumerate(zip(names, x, y, strict=True))
        ],
    )
