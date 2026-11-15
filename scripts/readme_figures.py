"""The README's charts, drawn from results files, each for GitHub's light and dark themes.

Usage: python scripts/readme_figures.py [--out docs/media]

Writes <name>_light.png and <name>_dark.png with a transparent background for:
  fig_distance      typical distance error, with the drone's position known and with its own
                    navigation (SynDrone test, Mid-Air test)
  fig_error_bars    how often a 90 % error bar contains the true distance, against the target
  fig_lost_drone    time added to a search when one of four drones is lost, with 95 % intervals
  fig_height        time to cover the area and people found, by flying height
  fig_false_alarms  false alarms before the find: every sighting, and after the two-sighting rule
  fig_thermal       people found by the thermal and the colour detector, on thermal images and on
                    colour video (HIT-UAV test days, VisDrone validation)
No experiment is run; every value is read from a results file (listed in SOURCES).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerialrange.utils.config import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "results"
SOURCES = {
    "syndrone": "syndrone/test/eval_p10_v2_cal.json",
    "nav": "midair/test/navigation/ranging_msckf_final.json",
    "headline": "headline.json",
    "paired": "multi_drone/phase16_paired_test.json",
    "test16": "multi_drone/phase16_test.json",
    "height": "multi_drone/search_speed_by_height.json",
    "thermal_on_thermal": "thermal/detect_dfine_hituav_day_coco_day_test.json",
    "colour_on_thermal": "thermal/detect_visdrone_dfine_n_640_day_test.json",
    "thermal_on_colour": "visdrone/val/detect_dfine_hituav_day_coco.json",
    "colour_on_colour": "visdrone/val/detect_dfine_n_640.json",
}
THEMES = {  # main bar, quiet bar, text, secondary text, axis line
    "light": {
        "main": "#2a78d6",
        "quiet": "#9ec5f4",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "axis": "#c3c2b7",
        "grid": "#e1e0d9",
    },
    "dark": {
        "main": "#3987e5",
        "quiet": "#184f95",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "axis": "#4a4a46",
        "grid": "#2c2c2a",
    },
}
WIDTH = 8.4  # inches at 200 dpi


def load() -> dict:
    return {k: json.loads((RES / v).read_text(encoding="utf-8")) for k, v in SOURCES.items()}


def frame(ax: plt.Axes, th: dict, xaxis: bool = False) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(xaxis)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(length=0, colors=th["ink2"], labelsize=10.5)
    ax.set_facecolor("none")


def title(fig: plt.Figure, th: dict, head: str, sub: str) -> None:
    fig.text(0.012, 0.975, head, fontsize=14, fontweight="bold", color=th["ink"], va="top")
    fig.text(
        0.012, 0.975 - 0.34 / fig.get_figheight(), sub, fontsize=10.5, color=th["ink2"], va="top"
    )


def hbars(
    th: dict,
    rows: list[tuple[str, str, float, str, bool]],
    head: str,
    sub: str,
    xmax: float,
    whiskers: list[tuple[float, float] | None] | None = None,
) -> plt.Figure:
    """Horizontal bars: (label, small label under it, value, value text, emphasised)."""
    extra = 0.2 * sub.count("\n")
    h = 0.95 + extra + 0.78 * len(rows)
    fig, ax = plt.subplots(figsize=(WIDTH, h), dpi=200)
    fig.patch.set_alpha(0)
    ys = list(range(len(rows)))[::-1]
    for n, (y, (_, _, v, txt, strong)) in enumerate(zip(ys, rows, strict=True)):
        ax.barh(y, v, height=0.36, color=th["main"] if strong else th["quiet"])
        end = v
        if whiskers and whiskers[n]:
            lo, hi = whiskers[n]
            ax.plot([lo, hi], [y, y], color=th["ink"], lw=1.4, solid_capstyle="butt")
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.09, y + 0.09], color=th["ink"], lw=1.4)
            end = hi
        ax.text(
            end + xmax * 0.012, y, txt, va="center", fontsize=13, fontweight="bold", color=th["ink"]
        )
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    frame(ax, th)
    left = 0.43
    fig.subplots_adjust(left=left, right=0.985, top=1 - (0.75 + extra) / h, bottom=0.06 / h)
    for y, (lab, small, *_rest) in zip(ys, rows, strict=True):
        ty = ax.transData.transform((0, y))[1] / (h * 200)
        fig.text(
            left - 0.015,
            ty + (0.11 / h if small else 0),
            lab,
            ha="right",
            va="center",
            fontsize=11.5,
            color=th["ink"],
        )
        if small:
            fig.text(
                left - 0.015,
                ty - 0.13 / h,
                small,
                ha="right",
                va="center",
                fontsize=9.5,
                color=th["ink2"],
            )
    title(fig, th, head, sub)
    return fig


def fig_distance(d: dict, th: dict) -> plt.Figure:
    p = d["syndrone"]["person"]["pooled"]["range_rel_err_median_abs"]
    v = d["syndrone"]["vehicle"]["pooled"]["range_rel_err_median_abs"]
    n = d["nav"]["conditions"]["fused"]["terrain"]["median"]
    rows = [
        ("People", "rendered towns, drone position known", 100 * p, f"{p:.1%}", False),
        ("Vehicles", "rendered towns, drone position known", 100 * v, f"{v:.1%}", False),
        ("Ground points", "uneven terrain, the drone's own navigation", 100 * n, f"{n:.1%}", True),
    ]
    return hbars(
        th,
        rows,
        "Typical distance error",
        "Median error as a share of the distance. Two test sets, each run once.",
        100 * n * 1.25,
    )


def fig_lost_drone(d: dict, th: dict) -> plt.Figure:
    c = d["paired"]["comparisons"]
    a, b = c["sweep_lose_any_minus_no_loss"]["time"], c["lawnmower_lose_any_minus_no_loss"]["time"]
    rows = [
        (
            "Searching as a team",
            "the others take over the lost drone's ground",
            a["mean"],
            f"+{a['mean']:,.0f} s",
            True,
        ),
        (
            "Fixed lanes",
            "each drone keeps to its own strip",
            b["mean"],
            f"+{b['mean']:,.0f} s",
            False,
        ),
    ]
    return hbars(
        th,
        rows,
        "Time added to a search when one of four drones is lost",
        "Seconds, averaged over 300 simulated searches each. Lines show 95% intervals.",
        b["ci95"][1] * 1.22,
        whiskers=[tuple(a["ci95"]), tuple(b["ci95"])],
    )


def fig_false_alarms(d: dict, th: dict) -> plt.Figure:
    r = d["test16"]["results"]["sweep_n4"]
    a, b = r["false_alarms_before_find_mean"], r["crew_alarms_before_find_mean"]
    rows = [
        ("Every sighting", "false alarms before the person is found", a, f"{a:,.0f}", False),
        ("After the two-sighting rule", "what the crew is asked to check", b, f"{b:,.0f}", True),
    ]
    return hbars(
        th,
        rows,
        "False alarms before the missing person is found",
        "Average per simulated search with four drones.",
        a * 1.18,
    )


def fig_error_bars(d: dict, th: dict) -> plt.Figure:
    m = d["headline"]["midair_test"]
    s = d["syndrone"]
    rows = [
        ("Flights over uneven terrain", m["cov_cal_90"], m["cov_raw_90"]),
        ("People, rendered towns", s["person"]["pooled"]["range_calibrated_coverage_0.90"], None),
        (
            "Vehicles, rendered towns",
            s["vehicle"]["pooled"]["range_calibrated_coverage_0.90"],
            None,
        ),
    ]
    h = 1.15 + 0.62 * len(rows) + 0.35
    fig, ax = plt.subplots(figsize=(WIDTH, h), dpi=200)
    fig.patch.set_alpha(0)
    ax.axvline(90, color=th["ink"], lw=1.2)
    ax.text(
        90.5,
        len(rows) - 0.42,
        "target: 90%",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=th["ink"],
    )
    for i, (_, v, raw) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([60, 100], [y, y], color=th["grid"], lw=1, zorder=1)
        if raw is not None:
            ax.annotate(
                "",
                xy=(100 * v - 0.9, y),
                xytext=(100 * raw + 0.9, y),
                arrowprops={"arrowstyle": "->", "color": th["ink2"], "lw": 1.2},
            )
            ax.scatter(
                [100 * raw], [y], s=120, facecolors="none", edgecolors=th["ink2"], lw=1.6, zorder=3
            )
            ax.text(
                100 * raw,
                y - 0.36,
                f"{raw:.0%} before calibration",
                ha="center",
                va="top",
                fontsize=9.5,
                color=th["ink2"],
            )
        ax.scatter([100 * v], [y], s=150, color=th["main"], zorder=4)
        ax.text(
            100 * v,
            y + 0.2,
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=th["ink"],
        )
    ax.set_xlim(60, 100)
    ax.set_ylim(-0.75, len(rows) - 0.1)
    ax.set_xticks([60, 70, 80, 90, 100], [f"{x}%" for x in (60, 70, 80, 90, 100)])
    ax.set_yticks([])
    frame(ax, th, xaxis=True)
    left = 0.36
    fig.subplots_adjust(left=left, right=0.975, top=1 - 0.95 / h, bottom=0.42 / h)
    for i, (lab, *_r) in enumerate(rows):
        ty = ax.transData.transform((0, len(rows) - 1 - i))[1] / (h * 200)
        fig.text(left - 0.02, ty, lab, ha="right", va="center", fontsize=11.5, color=th["ink"])
    title(
        fig,
        th,
        "How often a 90% error bar contains the true distance",
        "Share of distances whose error bar held, on test data run once.",
    )
    return fig


def fig_height(d: dict, th: dict) -> plt.Figure:
    r = d["height"]
    hs = sorted(r["by_height"].values(), key=lambda x: x["height_m"])
    labels = [f"{x['height_m']} m" for x in hs]
    cols = [th["main"] if x["height_m"] == r["today_m"] else th["quiet"] for x in hs]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9), dpi=200)
    fig.patch.set_alpha(0)
    panels = [
        (
            [x["sweep_min"] for x in hs],
            f"Minutes for {r['n_drones']} drones to cover the area once",
            lambda v: f"{v:.0f} min",
        ),
        (
            [100 * x["person_found_share"] for x in hs],
            "People the camera finds as it passes over",
            lambda v: f"{v:.0f}%" if v >= 1 else f"{v:.1f}%",
        ),
    ]
    for ax, (vals, head, fmt) in zip(axes, panels, strict=True):
        bars = ax.bar(labels, vals, width=0.5, color=cols)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.025,
                fmt(v),
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=th["ink"],
            )
        ax.set_title(head, loc="left", fontsize=11, color=th["ink"], pad=10)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_yticks([])
        frame(ax, th, xaxis=True)
        ax.tick_params(axis="x", labelsize=11, pad=5)
        ax.set_xlabel("flying height", fontsize=10, color=th["ink2"], labelpad=4)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.70, bottom=0.17, wspace=0.12)
    title(
        fig,
        th,
        "Flying higher would search faster, but the camera stops seeing people",
        "Highlighted bar: the height flown today. People found: the real detector on rendered "
        f"test towns. Area {r['area_km2']:.1f} km².",
    )
    return fig


def fig_thermal(d: dict, th: dict) -> plt.Figure:
    def thermal(k: str) -> tuple[float, tuple[float, float]]:
        g = d[k]["groups"]["person"]
        return g["ap50"]["ap"], tuple(g["ap50_ci95_resample_settings"])

    tt, tt_ci = thermal("thermal_on_thermal")
    ct, ct_ci = thermal("colour_on_thermal")
    cc = d["colour_on_colour"]["results"]["person"]["ap50"]["ap"]
    tc = d["thermal_on_colour"]["results"]["person"]["ap50"]["ap"]

    def fmt(v: float) -> str:
        return f"{v:.2f}" if v >= 0.01 else f"{v:.3f}"

    rows = [
        ("Thermal detector", "on thermal images", tt, fmt(tt), True),
        ("Colour detector", "on the same thermal images", ct, fmt(ct), False),
        ("Colour detector", "on colour video", cc, fmt(cc), True),
        ("Thermal detector", "on the same colour video", tc, fmt(tc), False),
    ]
    return hbars(
        th,
        rows,
        "Each detector finds people only with the camera it was trained for",
        "Person AP50 (1 means every person found and nothing else boxed). Thermal: two unseen\n"
        "HIT-UAV test days, lines show 95% intervals. Colour: VisDrone validation video.",
        1.0,
        whiskers=[tt_ci, ct_ci, None, None],
    )


FIGURES = {
    "fig_distance": fig_distance,
    "fig_error_bars": fig_error_bars,
    "fig_lost_drone": fig_lost_drone,
    "fig_height": fig_height,
    "fig_false_alarms": fig_false_alarms,
    "fig_thermal": fig_thermal,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/media")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    d = load()
    for name, fn in FIGURES.items():
        for mode, th in THEMES.items():
            fig = fn(d, th)
            fig.savefig(out / f"{name}_{mode}.png", transparent=True)
            plt.close(fig)
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
