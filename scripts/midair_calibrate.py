"""Fit on the calibration split, apply to the test split.

Usage:
  python scripts/midair_calibrate.py --level mid --base oracle_plane
  python scripts/midair_calibrate.py --level mid --base oracle_plane --apply-test --final

Fit (calibration split only):
  - one recalibration scalar s for sigma (maximum likelihood, e ~ N(0, (s sigma)^2));
  - the reliability-flag threshold on recalibrated relative sigma (`{{REL_UNC_THRESHOLD}}`),
    chosen by maximum F1 for "large error" = |relative error| > `{{ERR_THRESHOLD_FRAC}}`.
Limits (development split, where the curves are built):
  - `{{MIN_DEPRESSION_DEG}}` and `{{MAX_RANGE_M}}`: edges where the binned median |relative
    error| (noisy mode) has its upper 95 % bound at or above `{{ERR_THRESHOLD_FRAC}}`.
Apply (test split, once, --final): coverage and z spread before and after recalibration,
flag precision, recall and AUROC at the fixed threshold.
Writes results/midair/calibration_fit.json and, with --apply-test, results/midair/test/
calibration_applied.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import numpy as np

from aerialrange.data.midair import WATER_CLASS
from aerialrange.evaluate import midair_analysis as ana
from aerialrange.evaluate import stats as st
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import load_split_cache, noise_levels  # noqa: E402

COVERAGE_LEVELS = (0.5, 0.68, 0.9, 0.95, 0.99)
EDGES_DEP = [0, 2, 5, 10, 20, 30, 45, 60, 90]
EDGES_RANGE = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]


def noisy(split: str, sig: dict, base: str, seed: int) -> tuple[dict, dict, np.ndarray]:
    r, names = load_split_cache(split)
    nm = ana.noisy_mode(r, sig, base, np.random.default_rng([seed, 99]))
    ok = (
        (r["cls"] != WATER_CLASS)
        & nm["valid"]
        & np.isfinite(nm["pred_range_sigma_m"])
        & (nm["pred_range_sigma_m"] > 0)
    )
    return r, nm, ok


def calib_block(nm: dict, ok: np.ndarray, s: float) -> dict:
    e, sig = nm["range_err_m"][ok], nm["pred_range_sigma_m"][ok]
    out = {}
    for label, scale in (("before", 1.0), ("after", s)):
        z = e / (scale * sig)
        out[label] = {
            "z_std": float(np.std(z)),
            "coverage": st.coverage(z, COVERAGE_LEVELS),
            "mean_halfwidth_95_m": float(np.mean(st.z_halfwidth(0.95) * scale * sig)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="mid")
    ap.add_argument("--base", default="oracle_plane", choices=sorted(ana.BASES))
    ap.add_argument("--apply-test", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    if args.apply_test and not args.final:
        print("The test split runs once, after all settings are frozen. Use --final.")
        return 1
    unc, data = load_config("uncertainty"), load_config("data")
    seed = load_config("project").require("seed")
    err_thr = unc.require("err_threshold_frac")
    n_boot = unc.require("n_boot")
    min_n, min_g = data.require("min_samples_per_bin"), data.require("min_groups_per_bin")
    sig = noise_levels()[args.level]
    rng = np.random.default_rng(seed)

    # Fit on calibration.
    r_c, nm_c, ok_c = noisy("calibration", sig, args.base, seed)
    s = st.fit_sigma_scale(nm_c["range_err_m"][ok_c], nm_c["pred_range_sigma_m"][ok_c])
    score_c = s * nm_c["pred_range_sigma_m"][ok_c] / nm_c["range_est_m"][ok_c]
    large_c = np.abs(nm_c["rel_range_err"][ok_c]) > err_thr
    op = st.threshold_max_f1(score_c, large_c)

    # Limits on development.
    r_d, nm_d, ok_d = noisy("development", sig, args.base, seed)
    rel_d = np.abs(nm_d["rel_range_err"])
    dep_rows = st.binned_stat(
        r_d["true_depression_deg"][ok_d],
        rel_d[ok_d],
        r_d["group"][ok_d],
        EDGES_DEP,
        np.median,
        n_boot,
        min_n,
        rng,
        min_groups=min_g,
    )
    rng_rows = st.binned_stat(
        r_d["true_range_m"][ok_d],
        rel_d[ok_d],
        r_d["group"][ok_d],
        EDGES_RANGE,
        np.median,
        n_boot,
        min_n,
        rng,
        min_groups=min_g,
    )
    fit = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "level": args.level,
        "base": args.base,
        "sigmas_internal_units": sig,
        "err_threshold_frac": err_thr,
        "sigma_scale": s,
        "calibration_split": calib_block(nm_c, ok_c, s),
        "flag_operating_point": {
            **op,
            "score": "recalibrated sigma / estimated range",
            "rule": "max F1 on calibration split",
        },
        "flag_auroc_calibration": st.auroc(score_c, large_c),
        "limits_development": {
            "min_depression_deg": st.usable_limit(dep_rows, err_thr, "above"),
            "max_range_m": st.usable_limit(rng_rows, err_thr, "below"),
            "rule": "upper 95 % bound of binned median |relative error| below err_threshold_frac",
            "depression_bins": dep_rows,
            "range_bins": rng_rows,
        },
    }
    dest = REPO_ROOT / "results" / "midair" / "calibration_fit.json"
    dest.write_text(json.dumps(fit, indent=1, default=float) + "\n")
    print(
        json.dumps(
            {k: fit[k] for k in ("sigma_scale", "flag_operating_point", "flag_auroc_calibration")},
            indent=1,
            default=float,
        )
    )
    print(
        "limits:", {k: fit["limits_development"][k] for k in ("min_depression_deg", "max_range_m")}
    )

    if args.apply_test:
        r_t, nm_t, ok_t = noisy("test", sig, args.base, seed)
        score_t = s * nm_t["pred_range_sigma_m"][ok_t] / nm_t["range_est_m"][ok_t]
        large_t = np.abs(nm_t["rel_range_err"][ok_t]) > err_thr
        flagged = score_t >= op["threshold"]
        tp = int((flagged & large_t).sum())
        applied = {
            "fit": str(dest.relative_to(REPO_ROOT)),
            "test_split": calib_block(nm_t, ok_t, s),
            "flag_at_fixed_threshold": {
                "threshold": op["threshold"],
                "precision": tp / max(int(flagged.sum()), 1),
                "recall": tp / max(int(large_t.sum()), 1),
                "flagged_fraction": float(flagged.mean()),
            },
            "flag_auroc_test": st.auroc(score_t, large_t),
        }
        out = REPO_ROOT / "results" / "midair" / "test" / "calibration_applied.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(applied, indent=1, default=float) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
