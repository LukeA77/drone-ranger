"""Mid-Air analysis for one split, from the cache only.

Usage: python scripts/midair_analyse.py --split development [--final]

Writes to results/midair/<split>/:
  summary.json       base modes with trajectory-bootstrap intervals; water reported apart
  binned/*.csv       error by depression, range, slope, height above plane (both base modes)
  sweep.json         noisy-inputs mode per noise level and base: accuracy and calibration
                     (z spread, coverage at 50/68/90/95/99 %, sharpness)
  propagation.json   H2: noise-induced error versus first-order and Monte Carlo predictions
  sensitivity.json   tornado: per-input range-sigma contribution
  slope.json         H4: flat oracle versus tilted-plane and local-slope planes, by slope
  flag.json          reliability score (predicted relative sigma): AUROC and PR curve per
                     large-error definition
  figures/*.png      with a CSV table beside each
Calibration fitting (one scalar) and the flag operating point are fitted on the calibration
split by scripts/midair_calibrate.py, never here. The test split needs --final.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aerialrange.data.midair import WATER_CLASS
from aerialrange.evaluate import midair_analysis as ana
from aerialrange.evaluate import stats as st
from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz.plots import bar_chart, line_chart, write_csv

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import load_split_cache, noise_levels  # noqa: E402

BASES = ("oracle_plane", "relative_altitude")
COVERAGE_LEVELS = (0.5, 0.68, 0.9, 0.95, 0.99)
# Bin edges (degrees, metres). Chosen to resolve the steep change near the horizon and to be
# roughly logarithmic in range; bins below the minimum sample rule are not reported.
EDGES = {
    "true_depression_deg": [0, 2, 5, 10, 20, 30, 45, 60, 90],
    "true_range_m": [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000],
    "slope_deg": [0, 5, 10, 15, 20, 30, 45, 90],
    "h_oracle_m": [0, 5, 10, 20, 40, 80, 200],
}
LABELS = {
    "true_depression_deg": "True depression angle (degrees)",
    "true_range_m": "True range (m)",
    "slope_deg": "Local slope (degrees)",
    "h_oracle_m": "Height above oracle plane (m)",
}
MAD_TO_SIGMA = 0.6745  # median |x| of a zero-mean Gaussian is 0.6745 sigma
INPUT_LABELS = {
    "u": "Pixel u",
    "v": "Pixel v",
    "h": "Height above plane",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
    "tilt": "Camera tilt",
    "fx": "Focal length fx",
    "fy": "Focal length fy",
}


def p90(x: np.ndarray) -> float:
    return float(np.percentile(x, 90))


def summary_with_ci(
    r: dict, mask: np.ndarray, base_keys: dict, n_boot: int, rng: np.random.Generator
) -> dict:
    """Valid fraction and error statistics with trajectory-bootstrap 95 % intervals."""
    out: dict[str, Any] = {}
    valid = base_keys["valid"][mask]
    g_all = r["group"][mask]
    out["n_samples"] = int(mask.sum())
    out["valid_fraction"] = float(valid.mean())
    out["valid_fraction_ci"] = st.group_bootstrap_ci(
        valid.astype(float), g_all, np.mean, n_boot, rng
    )
    ok = mask & base_keys["valid"]
    g = r["group"][ok]
    rel = np.abs(base_keys["rel_range_err"][ok])
    a = np.abs(base_keys["range_err_m"][ok])
    hz = base_keys["horiz_err_m"][ok]
    for name, x, stat in (
        ("abs_rel_err_median", rel, np.median),
        ("abs_rel_err_p90", rel, p90),
        ("abs_range_err_median_m", a, np.median),
        ("abs_range_err_p95_m", a, lambda v: float(np.percentile(v, 95))),
        ("range_rmse_m", base_keys["range_err_m"][ok], lambda v: float(np.sqrt(np.mean(v**2)))),
        ("signed_range_err_median_m", base_keys["range_err_m"][ok], np.median),
        ("horiz_err_median_m", hz, np.median),
    ):
        out[name] = float(stat(x))
        out[name + "_ci"] = st.group_bootstrap_ci(x, g, stat, n_boot, rng)
    return out


def base_view(r: dict, base: str) -> dict:
    return {k: r[f"{base}.{k}"] for k in ("valid", "rel_range_err", "range_err_m", "horiz_err_m")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["development", "calibration", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument(
        "--mc-per-bin",
        type=int,
        default=100,
        help="samples per depression bin for the Monte Carlo prediction",
    )
    # Smoke-test hooks (tests/test_midair_analyse_smoke.py): alternative inputs and output, and
    # small resample counts. Overrides are refused when writing to results/.
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--splits-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-mc", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        print("The test split runs once, after all settings are frozen. Use --final.")
        return 1
    overrides = (args.n_boot, args.n_mc, args.min_samples)
    if any(v is not None for v in overrides) and args.out_dir is None:
        print("--n-boot/--n-mc/--min-samples are for smoke tests only; give --out-dir.")
        return 1
    data, unc = load_config("data"), load_config("uncertainty")
    seed = load_config("project").require("seed")
    n_boot = args.n_boot or unc.require("n_boot")
    n_mc = args.n_mc or unc.require("n_mc")
    min_n = args.min_samples or data.require("min_samples_per_bin")
    min_g = data.require("min_groups_per_bin")
    levels = noise_levels()
    rng = np.random.default_rng(seed)
    cache_kw = {
        k: v
        for k, v in (("cache_dir", args.cache_dir), ("splits_file", args.splits_file))
        if v is not None
    }
    r, names = load_split_cache(args.split, **cache_kw)
    out_dir = args.out_dir or REPO_ROOT / "results" / "midair" / args.split
    fig_dir = out_dir / "figures"
    land = r["cls"] != WATER_CLASS
    water = ~land
    note = (
        f"Mid-Air {args.split} split, {len(names)} trajectories; 95 % trajectory-bootstrap "
        f"intervals; bins need >= {min_n} samples and >= {min_g} trajectories."
    )

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "seed": seed,
        "split": args.split,
        "trajectories": names,
        "n_boot": n_boot,
        "n_mc": n_mc,
        "min_samples_per_bin": min_n,
        "min_groups_per_bin": min_g,
        "config_hash": {"data": data.hash(), "uncertainty": unc.hash()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(fname: str, payload: Any) -> None:
        """Write one stage's results as soon as it finishes (a late failure loses nothing)."""
        (out_dir / fname).write_text(
            json.dumps({"meta": meta, "data": payload}, indent=1, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {fname}", flush=True)

    # --- A. base modes ------------------------------------------------------------------
    summary = {
        b: {
            "land": summary_with_ci(r, land, base_view(r, b), n_boot, rng),
            "water": summary_with_ci(r, water, base_view(r, b), n_boot, rng)
            if water.sum() >= min_n
            else None,
        }
        for b in BASES
    }

    save("summary.json", summary)

    # --- B. binned curves ---------------------------------------------------------------
    binned: dict[str, dict] = {}
    for xkey, edges in EDGES.items():
        binned[xkey] = {}
        for b in BASES:
            ok = land & r[f"{b}.valid"]
            rows = st.binned_stat(
                r[xkey][ok],
                np.abs(r[f"{b}.rel_range_err"][ok]),
                r["group"][ok],
                edges,
                np.median,
                n_boot,
                min_n,
                rng,
                min_groups=min_g,
            )
            binned[xkey][b] = rows
            write_csv(out_dir / "binned" / f"{xkey}__{b}.csv", rows)
        series = {}
        for b in BASES:
            rows = [x for x in binned[xkey][b] if x["reported"]]
            series[b] = {
                "x": [(x["lo"] + x["hi"]) / 2 for x in rows],
                "y": [x["stat"] for x in rows],
                "lo": [x["ci_lo"] for x in rows],
                "hi": [x["ci_hi"] for x in rows],
            }
        line_chart(
            fig_dir / f"error_vs_{xkey}.png",
            series,
            f"Median absolute relative range error by {LABELS[xkey].split(' (')[0].lower()}",
            LABELS[xkey],
            "Median |relative error|",
            logx=xkey in ("true_range_m",),
            logy=True,
            note=note,
        )

    save("binned.json", binned)

    # --- C. noise sweep: accuracy and calibration ---------------------------------------
    sweep: dict[str, dict] = {}
    noisy_cache: dict[tuple[str, str], dict] = {}
    for li, (lname, sig) in enumerate(levels.items()):
        sweep[lname] = {"sigmas_internal_units": sig}
        for b in BASES:
            nm = ana.noisy_mode(r, sig, b, np.random.default_rng([seed, li]))
            noisy_cache[(lname, b)] = nm
            ok = (
                land
                & nm["valid"]
                & np.isfinite(nm["pred_range_sigma_m"])
                & (nm["pred_range_sigma_m"] > 0)
            )
            z = nm["range_err_m"][ok] / nm["pred_range_sigma_m"][ok]
            g = r["group"][ok]
            sweep[lname][b] = {
                "accuracy": summary_with_ci(
                    r, land, {k: nm[k] for k in base_view(r, b)}, n_boot, rng
                ),
                "z_std": float(np.std(z)),
                "z_std_ci": st.group_bootstrap_ci(z, g, np.std, n_boot, rng),
                "z_mad_sigma": float(np.median(np.abs(z)) / MAD_TO_SIGMA),
                "coverage": st.coverage(z, COVERAGE_LEVELS),
                "mean_halfwidth_95_m": float(
                    np.mean(st.z_halfwidth(0.95) * nm["pred_range_sigma_m"][ok])
                ),
                "scale_ml_this_split": st.fit_sigma_scale(
                    nm["range_err_m"][ok], nm["pred_range_sigma_m"][ok]
                ),
            }
    # Reliability diagram for the middle level, oracle base.
    mid = list(levels)[1]
    rel_series = {}
    for b in BASES:
        cov = sweep[mid][b]["coverage"]
        rel_series[b] = {
            "x": [100 * p for p in COVERAGE_LEVELS],
            "y": [100 * cov[p] for p in COVERAGE_LEVELS],
        }
    rel_series["ideal"] = {
        "x": [100 * p for p in COVERAGE_LEVELS],
        "y": [100 * p for p in COVERAGE_LEVELS],
    }
    line_chart(
        fig_dir / "reliability.png",
        rel_series,
        f"Interval coverage, noise level '{mid}' (before recalibration)",
        "Nominal coverage (%)",
        "Empirical coverage (%)",
        note=note,
    )

    save("sweep.json", sweep)

    # --- D. H2: noise-induced error versus first-order and Monte Carlo predictions -------
    nm = noisy_cache[(mid, "oracle_plane")]
    both = land & nm["valid"] & r["oracle_plane.valid"]
    induced = np.abs(nm["range_est_m"] - r["oracle_plane.range_est_m"])
    prop_rows = []
    edges = EDGES["true_depression_deg"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = both & (r["true_depression_deg"] >= lo) & (r["true_depression_deg"] < hi)
        ii = np.nonzero(m)[0]
        row: dict[str, Any] = {
            "lo": lo,
            "hi": hi,
            "n": int(len(ii)),
            "n_groups": int(len(np.unique(r["group"][ii]))),
        }
        if len(ii) >= min_n and row["n_groups"] >= min_g:
            row["empirical_median_abs_m"] = float(np.median(induced[ii]))
            row["empirical_ci"] = st.group_bootstrap_ci(
                induced[ii], r["group"][ii], np.median, n_boot, rng
            )
            sub = np.sort(rng.choice(ii, size=min(args.mc_per_bin, len(ii)), replace=False))
            fo = ana.first_order_sigma(r, sub, levels[mid], "oracle_plane")
            mc = ana.monte_carlo_sigma(
                r, sub, levels[mid], "oracle_plane", n_mc, np.random.default_rng([seed, lo])
            )
            row["first_order_pred_median_abs_m"] = float(MAD_TO_SIGMA * np.nanmedian(fo))
            row["monte_carlo_pred_median_abs_m"] = float(MAD_TO_SIGMA * np.nanmedian(mc))
            row["mc_over_fo_median"] = float(np.nanmedian(mc / fo))
            row["reported"] = True
        else:
            row["reported"] = False
        prop_rows.append(row)
    rep = [x for x in prop_rows if x["reported"]]
    xs = [(x["lo"] + x["hi"]) / 2 for x in rep]
    line_chart(
        fig_dir / "propagation_vs_depression.png",
        {
            "empirical (noise-induced)": {
                "x": xs,
                "y": [x["empirical_median_abs_m"] for x in rep],
                "lo": [x["empirical_ci"][0] for x in rep],
                "hi": [x["empirical_ci"][1] for x in rep],
            },
            "first-order prediction": {
                "x": xs,
                "y": [x["first_order_pred_median_abs_m"] for x in rep],
            },
            "Monte Carlo prediction": {
                "x": xs,
                "y": [x["monte_carlo_pred_median_abs_m"] for x in rep],
            },
        },
        f"Noise-induced range error, level '{mid}', oracle plane",
        LABELS["true_depression_deg"],
        "Median |range change| (m)",
        logy=True,
        note=note,
    )

    save("propagation.json", prop_rows)

    # --- E. sensitivity (tornado) --------------------------------------------------------
    sub = np.sort(
        rng.choice(
            np.nonzero(land & r["oracle_plane.valid"])[0],
            size=min(5000, int((land & r["oracle_plane.valid"]).sum())),
            replace=False,
        )
    )
    rs = {k: v[sub] for k, v in r.items()}
    sens = ana.sensitivity(rs, levels[mid], "oracle_plane")
    sens_out = {p: {k: v for k, v in d.items() if k != "per_sample"} for p, d in sens.items()}
    bar_chart(
        fig_dir / "tornado.png",
        [INPUT_LABELS[p] for p in sens_out],
        [sens_out[p]["median_m"] for p in sens_out],
        f"Range-sigma contribution per input, noise level '{mid}'",
        "First-order range sigma, median over samples (m)",
        note=f"Mid-Air {args.split} split, {len(names)} trajectories, {len(sub)} land samples; "
        "oracle-plane height; each input alone (contributions add in quadrature).",
    )

    save("sensitivity.json", sens_out)

    # --- F. slope experiment (H4) ---------------------------------------------------------
    tilted, local = ana.tilted_plane_mode(r), ana.local_slope_mode(r)
    slope_out = {}
    series = {}
    for name, view in (
        ("flat oracle", base_view(r, "oracle_plane")),
        ("tilted plane", tilted),
        ("local slope", local),
    ):
        ok = land & view["valid"]
        rows = st.binned_stat(
            r["slope_deg"][ok],
            np.abs(view["rel_range_err"][ok]),
            r["group"][ok],
            EDGES["slope_deg"],
            np.median,
            n_boot,
            min_n,
            rng,
            min_groups=min_g,
        )
        slope_out[name] = {"valid_fraction": float(view["valid"][land].mean()), "bins": rows}
        rp = [x for x in rows if x["reported"]]
        series[name] = {
            "x": [(x["lo"] + x["hi"]) / 2 for x in rp],
            "y": [x["stat"] for x in rp],
            "lo": [x["ci_lo"] for x in rp],
            "hi": [x["ci_hi"] for x in rp],
        }
    line_chart(
        fig_dir / "slope_planes.png",
        series,
        "Median |relative range error| by local slope, by plane model",
        LABELS["slope_deg"],
        "Median |relative error|",
        logy=True,
        note=note,
    )

    save("slope.json", slope_out)

    # --- G. reliability flag: AUROC and PR per large-error definition --------------------
    flag_out = {}
    nm = noisy_cache[(mid, "oracle_plane")]
    ok = land & nm["valid"] & np.isfinite(nm["pred_range_sigma_m"])
    score = nm["pred_range_sigma_m"][ok] / nm["range_est_m"][ok]
    for thr in unc.require("err_threshold_sweep"):
        label = np.abs(nm["rel_range_err"][ok]) > thr
        pr = st.precision_recall(score, label)
        keep = np.unique(np.linspace(0, len(pr["threshold"]) - 1, 200).astype(int))
        flag_out[str(thr)] = {
            "positive_fraction": float(label.mean()),
            "auroc": st.auroc(score, label),
            "pr_curve": {k: v[keep].tolist() for k, v in pr.items()},
        }
    flag_out["invalid_fraction_land"] = float(1 - nm["valid"][land].mean())

    save("flag.json", flag_out)
    print(f"wrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
