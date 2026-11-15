"""How flying height trades search speed against finding people (no new experiment).

Usage: python scripts/search_speed_by_height.py

For each SynDrone camera set-up (20, 50 and 80 m), combines two existing results:
- the time four drones take to cover the search area once: the simulation's own sweep time
  (area / (drones x speed x lane spacing), multidrone_run.scenario, frozen Phase 14 settings,
  the camera measured on the SynDrone test towns, sensor_p10_v2_cal.json);
- the share of people the real pipeline found from that height on the SynDrone test towns
  (results/syndrone/test/search_p10_v2_cal.json, run once in Phase 10).

Writes results/multi_drone/search_speed_by_height.json and
results/multi_drone/figures/search_speed_by_height.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import multidrone_run as mr  # noqa: E402
from aerialrange.search import sim  # noqa: E402
from aerialrange.utils.config import REPO_ROOT  # noqa: E402

SENSOR = REPO_ROOT / "results" / "multi_drone" / "sensor_p10_v2_cal.json"
SEARCH = REPO_ROOT / "results" / "syndrone" / "test" / "search_p10_v2_cal.json"
OUT = REPO_ROOT / "results" / "multi_drone" / "search_speed_by_height.json"
FIG = REPO_ROOT / "results" / "multi_drone" / "figures" / "search_speed_by_height.png"
HEIGHTS = (20, 50, 80)
N_DRONES = 4
TODAY_M = 20  # the height the search simulation flies (the only one where people are found)

INK, INK2, GRID, BLUE, BLUE_L = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#a9c9ef"


def compute() -> dict:
    st = mr.load_settings()
    found = json.loads(SEARCH.read_text(encoding="utf-8"))["person"]["by_height"]
    rows = {}
    for h in HEIGHTS:
        sc = mr.scenario(st, SENSOR, h, N_DRONES, "ring")
        sweep_s = sc.horizon_s / mr.HORIZON_COVERS
        rows[f"h{h}"] = {
            "height_m": h,
            "lane_spacing_m": sim.lane_spacing_m(sc.sensor),
            "sweep_s": sweep_s,
            "sweep_min": sweep_s / 60.0,
            "person_found_share": found[f"h{h}"]["found_share"],
        }
    sc20 = mr.scenario(st, SENSOR, TODAY_M, N_DRONES, "ring")
    return {
        "sources": {
            "sensor": SENSOR.relative_to(REPO_ROOT).as_posix(),
            "people_found": SEARCH.relative_to(REPO_ROOT).as_posix(),
            "settings": "configs/multi_drone.yaml (frozen, Phase 14)",
        },
        "n_drones": N_DRONES,
        "speed_ms": st["speed_ms"],
        "area_km2": sc20.grid.size * sc20.grid.cell_area_m2 / 1e6,
        "today_m": TODAY_M,
        "by_height": rows,
        "sweep_ratio_20_to_80": rows["h20"]["sweep_s"] / rows["h80"]["sweep_s"],
    }


def plot(r: dict) -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2}
    )  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=200, facecolor="white")
    rows = [r["by_height"][f"h{h}"] for h in HEIGHTS]
    labels = [f"{h} m" for h in HEIGHTS]
    colours = [BLUE if h == r["today_m"] else BLUE_L for h in HEIGHTS]

    def panel(ax: plt.Axes, vals: list[float], title: str, fmt, top: float) -> None:  # noqa: ANN001
        bars = ax.bar(labels, vals, width=0.56, color=colours, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt(v), ha="center",
                    va="bottom", fontsize=12, fontweight="bold", color=INK)  # fmt: skip
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=8)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", length=0, labelsize=11, pad=6)
        ax.set_xlabel("flying height", fontsize=10, labelpad=6)

    mins = [x["sweep_min"] for x in rows]
    found = [100 * x["person_found_share"] for x in rows]
    panel(axes[0], mins, f"Minutes for {r['n_drones']} drones to search the area once",
          lambda v: f"{v:.0f} min", max(mins) * 1.18)  # fmt: skip
    panel(axes[1], found, "People the camera finds as it flies over",
          lambda v: f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", max(found) * 1.18)  # fmt: skip
    fig.suptitle(
        "Flying higher would search faster, but today's camera stops seeing people",
        x=0.02, ha="left", fontsize=13.5, fontweight="bold", y=0.975,
    )  # fmt: skip
    fig.text(
        0.02, 0.015,
        f"Darker bar: the height the drones fly today. Area {r['area_km2']:.1f} km², "
        f"{r['speed_ms']:.1f} m/s. People found: the real detector on rendered SynDrone test "
        "towns, run once.",
        fontsize=8.5, color=INK2,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=4)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    plt.close(fig)


def main() -> int:
    r = compute()
    OUT.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(r)
    print(OUT.relative_to(REPO_ROOT).as_posix(), FIG.relative_to(REPO_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
