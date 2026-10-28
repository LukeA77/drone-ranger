"""End-to-end ranging error budget for detected boxes (D-622).

Usage: python scripts/contact_point_budget.py

1. Contact-point error (simulated, exact truth; evaluate/contact_point.py): cars and people on
   flat ground in AU-AIR-like views (focal length 1,259 px, 1920 x 1080, heights 10 / 20 / 30 m,
   camera tilts 15 to 80 degrees), ranged from the box bottom-centre as the pipeline does, with
   and without the detector's measured box noise (configs/uncertainty.yaml sigma_pix_model,
   from VisDrone validation). The box centre is shown for comparison.
2. End to end: the contact-point error combined with the ranging error measured on the Mid-Air
   test flights (terrain from motion, exact poses; the paired samples of D-620), drawn
   independently: combined = (1 + ranging error) (1 + contact error) - 1. Independence is an
   assumption, stated.
Writes results/ranging/contact_point_budget.json and a figure.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np

from aerialrange.data.midair import WATER_CLASS
from aerialrange.evaluate.contact_point import PinholeView, simulate
from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz.plots import line_chart

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from terrain_analyse import load, methods  # noqa: E402

CLASSES = {  # dims (length, width, height) ranges in metres, and the noise group
    "car": ((3.8, 1.6, 1.4), (5.2, 2.0, 1.8), "vehicle"),
    "person": ((0.4, 0.3, 1.55), (0.6, 0.5, 1.85), "person"),
}
HEIGHTS_M = (10.0, 20.0, 30.0)
TILTS_DEG = (15.0, 30.0, 45.0, 60.0, 80.0)
N_PER_VIEW = 4000
DEP_EDGES = [0, 10, 20, 40, 60, 91]


def summary(rel: np.ndarray) -> dict:
    a = np.abs(rel)
    return {
        "n": int(len(rel)),
        "median_signed": float(np.median(rel)),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
    }


def main() -> int:
    seed = load_config("project").require("seed")
    f_px = float(load_config("auair").require("focal_px"))
    kappa = load_config("uncertainty").require("sigma_pix_model")
    rng = np.random.default_rng(seed)
    # ranging error on the paired Mid-Air test samples (D-620), signed
    r, _ = load("test")
    m = methods(r)
    land = r["cls"] != WATER_CLASS
    pair = land & m["terrain_motion"]["valid"] & m["flat_true_height"]["valid"]
    e_range = m["terrain_motion"]["rel"][pair]
    out: dict = {"classes": {}}
    for cls, (lo, hi, group) in CLASSES.items():
        k = kappa[group]
        res = {}
        for label, ku, kv, mode in (
            ("bottom_centre_no_box_noise", 0.0, 0.0, "bottom_centre"),
            ("bottom_centre_with_box_noise", k["kappa_u"], k["kappa_v"], "bottom_centre"),
            ("centre_with_box_noise", k["kappa_u"], k["kappa_v"], "centre"),
        ):
            rel, dep = [], []
            for h in HEIGHTS_M:
                for t in TILTS_DEG:
                    s = simulate(
                        PinholeView(f_px, 1920, 1080, h, t), N_PER_VIEW, lo, hi, ku, kv, rng, mode
                    )
                    rel.append(s["rel"])
                    dep.append(s["depression_deg"])
            rel, dep = np.concatenate(rel), np.concatenate(dep)
            bins = {}
            for a, b in zip(DEP_EDGES[:-1], DEP_EDGES[1:], strict=True):
                sel = (dep >= a) & (dep < b)
                if sel.sum() >= 100:
                    bins[f"{a}-{b}"] = summary(rel[sel])
            res[label] = {"all": summary(rel), "by_depression_deg": bins}
            if label == "bottom_centre_with_box_noise":
                er = rng.choice(e_range, len(rel))
                comb = (1 + er) * (1 + rel) - 1
                res["end_to_end_terrain_exact_poses"] = summary(comb)
        out["classes"][cls] = res
        b = res["bottom_centre_with_box_noise"]["all"]
        e2e = res["end_to_end_terrain_exact_poses"]
        print(
            f"{cls}: contact point median |error| {b['median_abs']:.1%} "
            f"(signed {b['median_signed']:+.1%}, p90 {b['p90_abs']:.1%}); "
            f"end to end {e2e['median_abs']:.1%} (p90 {e2e['p90_abs']:.1%})"
        )
    out["ranging_only_paired_median_abs"] = float(np.median(np.abs(e_range)))
    out["meta"] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-622",
        "seed": seed,
        "focal_px": f_px,
        "heights_m": HEIGHTS_M,
        "tilts_deg": TILTS_DEG,
        "n_per_view": N_PER_VIEW,
        "truth": "range to the centre of the object's footprint on the ground",
        "box_noise": "configs/uncertainty.yaml sigma_pix_model (VisDrone val, relative to box)",
        "end_to_end": "Mid-Air test paired terrain errors (exact poses) combined independently",
    }
    dest = REPO_ROOT / "results/ranging"
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    (dest / "contact_point_budget.json").write_text(json.dumps(out, indent=1) + "\n")
    series = {}
    for cls in CLASSES:
        bins = out["classes"][cls]["bottom_centre_with_box_noise"]["by_depression_deg"]
        mids = [(int(k.split("-")[0]) + int(k.split("-")[1])) / 2 for k in bins]
        series[f"{cls.capitalize()}, box bottom-centre"] = {
            "x": mids,
            "y": [v["median_abs"] for v in bins.values()],
        }
    line_chart(
        dest / "figures/contact_point_error.png",
        series,
        title="Error from ranging a detected box instead of a ground point",
        xlabel="Line of sight below the horizon (degrees)",
        ylabel="Median |relative range error|",
        logy=True,
        note="Simulated objects on flat ground, AU-AIR camera, measured detector box noise; "
        "exact poses. D-622.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
