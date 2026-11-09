"""Effect of fused poses on terrain-from-motion ranging (Phase 11, D-741).

Usage: python scripts/msckf_ranging.py --nav msckf [--split development] [--frame-step 1]
       [--workers 3]

Re-runs version 1's pose comparison (scripts/terrain_pose_sweep.py, D-621) on the same samples
with these pose conditions:
- `exact`: the simulator's poses;
- `nav_good`, `nav_typical`, `nav_poor`: version 1's assumed navigation grades, same seeds;
- `fused`: the filter's poses (scripts/msckf_midair.py --name <nav>), as a real-time system
  has them at frame k: the clone estimates at k and k - g from the filter's window, and the
  range uncertainty from the filter's own 12x12 joint covariance of the two poses
  (`triangulate_batch(pose_cov=...)`), not from an assumed grade;
- `fused_no_cov`: the same poses with no pose term in the uncertainty (shows what the filter's
  covariance contributes to coverage).
Only frames where the filter is running (after its alignment) are scored, for every condition.
Reports land samples: median and 90th percentile of |relative range error| with trajectory-
bootstrap intervals, valid fraction, and raw coverage of +-z sigma at 68 / 90 / 95 %.
Writes results/midair/<split>/navigation/ranging_<nav>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import norm

from aerialrange.data.midair import WATER_CLASS, MidAirTrajectory, midair_camera
from aerialrange.evaluate import stats as st
from aerialrange.evaluate.terrain_motion import points_from_tracks, range_targets, track_frame
from aerialrange.geometry.camera import Pose, pixel_to_world_ray
from aerialrange.navigation.run import REL_GAPS
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import CACHE, midair_root, split_names  # noqa: E402
from midair_terrain import motion_settings  # noqa: E402
from terrain_pose_sweep import CONDITIONS as SWEEP  # noqa: E402
from terrain_pose_sweep import perturb  # noqa: E402

NAV = REPO_ROOT / "cache" / "midair" / "navigation"
GRADES = ("exact", "nav_good", "nav_typical", "nav_poor")
CONDS = (*GRADES, "fused", "fused_no_cov")
LEVELS = (0.68, 0.90, 0.95)


def run_trajectory(root: Path, name: str, nav: str, frame_step: int, seed: int) -> dict:
    s = motion_settings()
    if tuple(s.frame_gaps) != REL_GAPS:
        raise ValueError(f"filter window gaps {REL_GAPS} differ from terrain gaps {s.frame_gaps}")
    r = dict(np.load(CACHE / f"{name}.npz"))
    fz = np.load(NAV / nav / f"{name}.npz")
    row_of = {int(k): i for i, k in enumerate(fz["frames"])}
    traj = MidAirTrajectory(root, name)
    n = len(r["u"])
    out = {c: {k: np.full(n, np.nan) for k in ("terrain", "sigma", "flat")} for c in CONDS}
    scored = np.zeros(n, bool)
    tnum = int(name.split("_")[-1])
    sig_by_gap = dict(zip(s.frame_gaps, s.sigma_px_by_gap, strict=True))
    keys = np.unique(r["frame"])[::frame_step]
    ci_of = {c: i for i, c in enumerate(SWEEP)}
    try:
        for k in keys:
            k = int(k)
            if k not in row_of:
                continue
            row = row_of[k]
            wr, wp, wc = fz["win_rot"][row], fz["win_pos"][row], fz["win_cov"][row]
            gaps = [
                g
                for j, g in enumerate(REL_GAPS)
                if k - g >= 0 and np.isfinite(wp[1 + j]).all() and np.isfinite(wc[j]).all()
            ]
            if not gaps or not np.isfinite(wp[0]).all():
                continue
            idx = np.nonzero(r["frame"] == k)[0]
            scored[idx] = True
            img_k = cv2.imread(str(traj.frame_path("color_left", k)), cv2.IMREAD_GRAYSCALE)
            prev_imgs = [
                cv2.imread(str(traj.frame_path("color_left", k - g)), cv2.IMREAD_GRAYSCALE)
                for g in gaps
            ]
            cam = midair_camera(img_k.shape[1], img_k.shape[0])
            uvk, tracks = track_frame(img_k, prev_imgs, s)
            uv = np.stack([r["u"][idx], r["v"][idx]], axis=1).astype(float)
            frames = [k] + [k - g for g in gaps]
            true = {f: traj.pose(f) for f in frames}
            for c in CONDS:
                if c in GRADES:
                    pj, aj, pb, ab = SWEEP[c]
                    ci = ci_of[c]  # same seeds as the version 1 sweep
                    rb = np.random.default_rng([seed, tnum, ci])
                    bp = rb.normal(0, pb, 3)
                    ba = rb.normal(0, np.radians(ab), 3)
                    poses = {}
                    for f in frames:
                        rj = np.random.default_rng([seed, tnum, ci, f])
                        poses[f] = perturb(
                            true[f], bp + rj.normal(0, pj, 3), ba + rj.normal(0, np.radians(aj), 3)
                        )
                    sc = replace(s, sigma_pos_m=float(pj), sigma_att_rad=float(np.radians(aj)))
                    prev = [(poses[k - g], sig_by_gap[g]) for g in gaps]
                    pose_k = poses[k]
                else:
                    j_of = {g: j for j, g in enumerate(REL_GAPS)}
                    pose_k = Pose(wp[0], wr[0])
                    sc = replace(s, sigma_pos_m=0.0, sigma_att_rad=0.0)
                    prev = [
                        (
                            Pose(wp[1 + j_of[g]], wr[1 + j_of[g]]),
                            sig_by_gap[g],
                            wc[j_of[g]] if c == "fused" else None,
                        )
                        for g in gaps
                    ]
                P, var_z, _, uv_pts = points_from_tracks(cam, pose_k, uvk, tracks, prev, sc)
                res = range_targets(cam, pose_k, uv, P, var_z, sc, uv_points=uv_pts)
                out[c]["terrain"][idx] = np.where(res["valid"], res["range_m"], np.nan)
                out[c]["sigma"][idx] = np.where(res["valid"], res["range_sigma_m"], np.nan)
                C, D = pixel_to_world_ray(cam, pose_k, uv)
                h = r["z_oracle"][idx] - C[2]
                lam = np.where((D[:, 2] > 0) & (h > 0), h / D[:, 2], np.nan)
                out[c]["flat"][idx] = lam * np.linalg.norm(D, axis=1)
    finally:
        traj.close()
    out["scored"] = scored
    return out


def summarise(
    est: np.ndarray,
    sig: np.ndarray | None,
    truth: np.ndarray,
    ok_mask: np.ndarray,
    g: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    rel = np.abs(est - truth) / truth
    ok = ok_mask & np.isfinite(rel)
    a = rel[ok]
    res = {
        "valid_fraction": float(ok.sum() / ok_mask.sum()),
        "median": float(np.median(a)),
        "median_ci": st.group_bootstrap_ci(a, g[ok], np.median, n_boot, rng),
        "p90": float(np.percentile(a, 90)),
    }
    if sig is not None:
        okz = ok & np.isfinite(sig) & (sig > 0)
        z = np.abs(est - truth)[okz] / sig[okz]
        res["coverage_raw"] = {
            f"{lv:.2f}": float(np.mean(z <= norm.ppf(0.5 + lv / 2))) for lv in LEVELS
        }
        res["z_rms"] = float(np.sqrt(np.mean(z**2)))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", required=True)
    ap.add_argument("--split", default="development")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    seed = load_config("project").require("seed")
    n_boot = load_config("uncertainty").require("n_boot")
    root = midair_root()
    names = [
        n
        for n in split_names(args.split)
        if (NAV / args.nav / f"{n}.npz").exists() and (CACHE / f"{n}.npz").exists()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        outs = list(
            ex.map(
                run_trajectory,
                [root] * len(names),
                names,
                [args.nav] * len(names),
                [args.frame_step] * len(names),
                [seed] * len(names),
            )
        )
    truth, land, groups = [], [], []
    est = {c: {k: [] for k in ("terrain", "sigma", "flat")} for c in CONDS}
    for name, o in zip(names, outs, strict=True):
        r = dict(np.load(CACHE / f"{name}.npz"))
        m = o["scored"]
        truth.append(r["true_range_m"][m])
        land.append((r["cls"] != WATER_CLASS)[m])
        groups.append(np.full(m.sum(), int(name.split("_")[-1])))
        for c in CONDS:
            for k in ("terrain", "sigma", "flat"):
                est[c][k].append(o[c][k][m])
    truth, land, groups = map(np.concatenate, (truth, land, groups))
    rng = np.random.default_rng(seed)
    results = {}
    for c in CONDS:
        e = {k: np.concatenate(v) for k, v in est[c].items()}
        results[c] = {
            "terrain": summarise(e["terrain"], e["sigma"], truth, land, groups, n_boot, rng),
            "flat": summarise(e["flat"], None, truth, land, groups, n_boot, rng),
        }
        t = results[c]["terrain"]
        print(
            f"{c:14s} terrain median {t['median']:.4f} p90 {t['p90']:.3f} valid "
            f"{t['valid_fraction']:.3f} cov {t['coverage_raw']}  "
            f"flat {results[c]['flat']['median']:.4f}"
        )
    rec = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "split": args.split,
        "nav_run": args.nav,
        "trajectories": names,
        "land_samples": int(land.sum()),
        "frame_step": args.frame_step,
        "seed": seed,
        "n_boot": n_boot,
        "nav_grades": "version 1's illustrative assumptions (D-621), same seeds",
        "metric": "land samples while the filter runs; |relative range error|; raw coverage "
        "of +-z sigma with the method's own sigma (no calibration factor)",
        "conditions": results,
    }
    dest = REPO_ROOT / "results" / "midair" / args.split / "navigation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"ranging_{args.nav}.json").write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
