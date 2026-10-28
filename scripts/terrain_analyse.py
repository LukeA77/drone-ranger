"""Compare flat-ground and terrain-from-motion ranging on Mid-Air, and calibrate (plans D, B, E).

Usage:
  python scripts/terrain_analyse.py --split development
  python scripts/terrain_analyse.py --fit-calibration          (fits B and E on calibration)
  python scripts/terrain_analyse.py --split test --final       (applies the frozen B and E)

Methods compared on the same samples (land only; water reported apart by midair_analyse):
  flat_true_height   flat plane at the true median ground height (oracle_plane; unrealistic)
  flat_altimeter     flat plane from a take-off-relative altimeter (relative_altitude)
  terrain_motion     plan D: needs no height above ground
Outputs in results/midair/<split>/terrain/: summary.json (bootstrap intervals), binned CSVs,
calibration.json (raw z and, with a fitted model, conformal coverage), flag.json, figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np

from aerialrange.data.midair import WATER_CLASS
from aerialrange.evaluate import stats as st
from aerialrange.evaluate.conformal import apply_conformal, fit_conformal
from aerialrange.evaluate.flag import FlagModel, fit_flag
from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz.plots import line_chart

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import CACHE, split_names  # noqa: E402

LEVELS = (0.5, 0.68, 0.9, 0.95, 0.99)
DEP_EDGES = [0, 2, 5, 10, 20, 30, 45, 60, 90]
SLOPE_EDGES = [0, 5, 10, 15, 20, 30, 45, 90]
FIT = REPO_ROOT / "results" / "midair" / "terrain_calibration_fit.json"
FLAG_FEATURES = ["rel_sigma", "roughness_over_range", "support_over_range", "log_n_points"]


def load(split: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Evaluation cache joined with terrain results, for trajectories that have both."""
    parts, names = [], []
    for name in split_names(split):
        a, b = CACHE / f"{name}.npz", CACHE / f"{name}.terrain.npz"
        if a.exists() and b.exists():
            d = dict(np.load(a))
            d.update(dict(np.load(b)))
            d["group"] = np.full(len(d["u"]), int(name.split("_")[-1]))
            parts.append(d)
            names.append(name)
    if not parts:
        raise FileNotFoundError(f"no terrain results for split {split!r}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, names


def methods(r: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    t = r["true_range_m"]
    out = {}
    for name, key in (
        ("flat_true_height", "oracle_plane"),
        ("flat_altimeter", "relative_altitude"),
    ):
        out[name] = {"valid": r[f"{key}.valid"], "range": r[f"{key}.range_est_m"]}
    out["terrain_motion"] = {
        "valid": r["terrain.valid"],
        "range": r["terrain.range_m"],
        "sigma": r["terrain.range_sigma_m"],
    }
    for m in out.values():
        m["rel"] = (m["range"] - t) / t
    return out


def flag_features(r: dict[str, np.ndarray], sigma: np.ndarray) -> np.ndarray:
    rng = r["terrain.range_m"]
    return np.column_stack(
        [
            sigma / rng,
            r["terrain.roughness_m"] / rng,
            r["terrain.support_radius_m"] / rng,
            np.log1p(r["terrain.n_points"]),
        ]
    )


def dep_band(r: dict[str, np.ndarray]) -> np.ndarray:
    """Viewing-angle band of the estimated ray (deployable: from the estimate, not truth)."""
    P, C = r["terrain.point_w"], r["pose_p"]
    dep = np.degrees(np.arctan2(P[:, 2] - C[:, 2], np.hypot(*(P[:, :2] - C[:, :2]).T)))
    return np.digitize(dep, DEP_EDGES[1:-1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["development", "calibration", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--fit-calibration", action="store_true")
    args = ap.parse_args(argv)
    unc, data = load_config("uncertainty"), load_config("data")
    seed = load_config("project").require("seed")
    n_boot = unc.require("n_boot")
    err_thr = unc.require("err_threshold_frac")
    min_n, min_g = data.require("min_samples_per_bin"), data.require("min_groups_per_bin")
    rng = np.random.default_rng(seed)

    if args.fit_calibration:
        r, names = load("calibration")
        land = r["cls"] != WATER_CLASS
        m = methods(r)["terrain_motion"]
        ok = land & m["valid"] & np.isfinite(m["sigma"]) & (m["sigma"] > 0)
        err = (m["range"] - r["true_range_m"])[ok]
        band = dep_band(r)[ok]
        model = fit_conformal(err, m["sigma"][ok], LEVELS, groups=band, min_group=min_n)
        sig68 = apply_conformal(m["sigma"][ok], model, 0.68, groups=band)
        X = flag_features({k: v[ok] for k, v in r.items()}, sig68)
        y = np.abs(m["rel"][ok]) > err_thr
        finite = np.all(np.isfinite(X), axis=1)
        l2 = unc.require("flag_l2")
        flag = fit_flag(X[finite], y[finite], FLAG_FEATURES, l2=l2)
        score = flag.score(X[finite])
        op = st.threshold_max_f1(score, y[finite])
        fit = {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "trajectories": names,
            "conformal": json.loads(json.dumps(model, default=str)),
            "flag_model": flag.to_dict(),
            "flag_threshold": op,
            "flag_auroc_calibration": st.auroc(score, y[finite]),
            "err_threshold_frac": err_thr,
        }
        FIT.write_text(json.dumps(fit, indent=1, default=float) + "\n")
        print(
            json.dumps(
                {k: fit[k] for k in ("flag_threshold", "flag_auroc_calibration")},
                indent=1,
                default=float,
            )
        )
        print("conformal global multipliers:", model["global"])
        return 0

    if args.split is None:
        ap.error("give --split or --fit-calibration")
    if args.split == "test" and not args.final:
        print("The test split runs once, after all settings are frozen. Use --final.")
        return 1
    r, names = load(args.split)
    land = r["cls"] != WATER_CLASS
    ms = methods(r)
    out_dir = REPO_ROOT / "results" / "midair" / args.split / "terrain"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    note = (
        f"Mid-Air {args.split} split, {len(names)} trajectories, land samples; 95 % "
        "trajectory-bootstrap intervals."
    )
    g = r["group"]

    summary: dict[str, Any] = {}
    for name, m in ms.items():
        ok = land & m["valid"]
        a = np.abs(m["rel"][ok])
        summary[name] = {
            "valid_fraction": float(m["valid"][land].mean()),
            "abs_rel_err_median": float(np.median(a)),
            "abs_rel_err_median_ci": st.group_bootstrap_ci(a, g[ok], np.median, n_boot, rng),
            "abs_rel_err_p90": float(np.percentile(a, 90)),
            "abs_rel_err_p90_ci": st.group_bootstrap_ci(
                a, g[ok], lambda v: float(np.percentile(v, 90)), n_boot, rng
            ),
        }
    binned: dict[str, Any] = {}
    for xkey, edges, label in (
        ("true_depression_deg", DEP_EDGES, "True depression angle"),
        ("slope_deg", SLOPE_EDGES, "Local slope"),
    ):
        binned[xkey] = {}
        series = {}
        for name, m in ms.items():
            ok = land & m["valid"]
            rows = st.binned_stat(
                r[xkey][ok],
                np.abs(m["rel"][ok]),
                g[ok],
                edges,
                np.median,
                n_boot,
                min_n,
                rng,
                min_groups=min_g,
            )
            binned[xkey][name] = rows
            rep = [x for x in rows if x["reported"]]
            series[name] = {
                "x": [(x["lo"] + x["hi"]) / 2 for x in rep],
                "y": [x["stat"] for x in rep],
                "lo": [x["ci_lo"] for x in rep],
                "hi": [x["ci_hi"] for x in rep],
            }
        line_chart(
            fig_dir / f"terrain_vs_flat_{xkey}.png",
            series,
            f"Median |relative range error| by {label.lower()}: flat vs terrain-aware",
            f"{label} (degrees)",
            "Median |relative error|",
            logy=True,
            note=note,
        )

    t = ms["terrain_motion"]
    ok = land & t["valid"] & np.isfinite(t["sigma"]) & (t["sigma"] > 0)
    z = (t["range"] - r["true_range_m"])[ok] / t["sigma"][ok]
    calib: dict[str, Any] = {
        "raw": {
            "z_std": float(np.std(z)),
            "z_robust_spread": float(np.median(np.abs(z)) / 0.6745),
            "coverage": st.coverage(z, LEVELS),
        }
    }
    flag_out: dict[str, Any] = {}
    if FIT.exists():
        fit = json.loads(FIT.read_text())
        model = fit["conformal"]
        model["global"] = {float(k): float(v) for k, v in model["global"].items()}
        model["groups"] = {
            int(k): {(float(kk) if kk != "n" else kk): vv for kk, vv in v.items()}
            for k, v in model["groups"].items()
        }
        band = dep_band(r)[ok]
        err = (t["range"] - r["true_range_m"])[ok]
        calib["conformal"] = {
            str(p): float(
                np.mean(np.abs(err) <= apply_conformal(t["sigma"][ok], model, p, groups=band))
            )
            for p in LEVELS
        }
        sig68 = apply_conformal(t["sigma"][ok], model, 0.68, groups=band)
        X = flag_features({k: v[ok] for k, v in r.items()}, sig68)
        y = np.abs(t["rel"][ok]) > err_thr
        fin = np.all(np.isfinite(X), axis=1)
        flag = FlagModel.from_dict(fit["flag_model"])
        score = flag.score(X[fin])
        flagged = score >= fit["flag_threshold"]["threshold"]
        tp = int((flagged & y[fin]).sum())
        flag_out = {
            "auroc": st.auroc(score, y[fin]),
            "precision": tp / max(int(flagged.sum()), 1),
            "recall": tp / max(int(y[fin].sum()), 1),
            "flagged_fraction": float(flagged.mean()),
            "coefficients": dict(zip(flag.names, flag.coef.tolist(), strict=True)),
        }
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "trajectories": names,
        "n_boot": n_boot,
        "seed": seed,
        "err_threshold_frac": err_thr,
    }
    for fname, payload in (
        ("summary.json", summary),
        ("binned.json", binned),
        ("calibration.json", calib),
        ("flag.json", flag_out),
    ):
        (out_dir / fname).write_text(
            json.dumps({"meta": meta, "data": payload}, indent=1, default=float) + "\n"
        )
    print(
        json.dumps(
            {
                k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                for k, v in summary.items()
            },
            indent=1,
        )
    )
    print(json.dumps(calib, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
