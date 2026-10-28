"""Pose-noise sensitivity of terrain from motion (D-621).

Usage: python scripts/terrain_pose_sweep.py [--frame-step 4] [--workers 4] [--check-exact]

The Mid-Air evaluation gives terrain from motion the simulator's exact poses. A real drone's
navigation has errors of two kinds, which affect it differently:
- jitter: independent error in every frame's pose. Two-view triangulation over 1 to 10 frames
  (0.04 to 0.4 s) depends on the change in pose between the two frames, so this matters most.
- bias: one error shared by every frame of a flight (a GNSS position offset, an attitude or
  mounting error). A shared position offset moves the camera and the triangulated ground
  together; a shared attitude error tilts every ray.
Each condition perturbs every pose the method uses, including the pose that casts the target
ray, exactly as a navigation error would. Feature tracks do not depend on the pose, so they are
computed once per frame and reused. The triangulation covariance is told the jitter level, as
a deployed system configured for its navigation grade would be. The flat-ground baseline
(true ground height, same noisy target pose) is scored on the same samples for comparison.

Development flights only (the test frames were released after the one-time test run). A
sensitivity analysis, not a tuning step: no setting is chosen from it.
Writes results/midair/development/terrain/pose_sweep.json and a figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from aerialrange.data.midair import WATER_CLASS, MidAirTrajectory, midair_camera
from aerialrange.evaluate import stats as st
from aerialrange.evaluate.terrain_motion import (
    MotionSettings,
    points_from_tracks,
    range_targets,
    track_frame,
)
from aerialrange.geometry.camera import Pose, pixel_to_world_ray
from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz.plots import line_chart

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import CACHE, split_names  # noqa: E402
from midair_terrain import motion_settings  # noqa: E402

# name: (position jitter m, attitude jitter deg, position bias m, attitude bias deg), per axis
CONDITIONS = {
    "exact": (0, 0, 0, 0),
    **{f"pos_jitter_{v:g}m": (v, 0, 0, 0) for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)},
    **{f"att_jitter_{v:g}deg": (0, v, 0, 0) for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)},
    **{f"pos_bias_{v:g}m": (0, 0, v, 0) for v in (1.0, 3.0, 5.0)},
    **{f"att_bias_{v:g}deg": (0, 0, 0, v) for v in (0.25, 0.5, 1.0, 2.0)},
    # illustrative navigation grades (assumptions, not a product specification)
    "nav_good": (0.02, 0.02, 1.0, 0.25),
    "nav_typical": (0.05, 0.05, 2.0, 0.5),
    "nav_poor": (0.2, 0.2, 5.0, 1.0),
}


def perturb(pose: Pose, dp: np.ndarray, rotvec: np.ndarray) -> Pose:
    """Pose with a position offset (NED, m) and a small world-frame rotation error."""
    return Pose(pose.p_w + dp, Rotation.from_rotvec(rotvec).as_matrix() @ pose.R_wb)


def run_trajectory(
    root: Path, name: str, s: MotionSettings, frame_step: int, seed: int
) -> dict[str, dict[str, np.ndarray]]:
    r = dict(np.load(CACHE / f"{name}.npz"))
    traj = MidAirTrajectory(root, name)
    n = len(r["u"])
    out = {c: {"terrain": np.full(n, np.nan), "flat": np.full(n, np.nan)} for c in CONDITIONS}
    tnum = int(name.split("_")[-1])
    sig_by_gap = dict(zip(s.frame_gaps, s.sigma_px_by_gap, strict=True))
    keys = np.unique(r["frame"])[::frame_step]
    try:
        for k in keys:
            idx = np.nonzero(r["frame"] == k)[0]
            gaps = [g for g in s.frame_gaps if k - g >= 0]
            if not gaps:
                continue
            img_k = cv2.imread(str(traj.frame_path("color_left", int(k))), cv2.IMREAD_GRAYSCALE)
            prev_imgs = [
                cv2.imread(str(traj.frame_path("color_left", int(k) - g)), cv2.IMREAD_GRAYSCALE)
                for g in gaps
            ]
            cam = midair_camera(img_k.shape[1], img_k.shape[0])
            uvk, tracks = track_frame(img_k, prev_imgs, s)
            uv = np.stack([r["u"][idx], r["v"][idx]], axis=1).astype(float)
            frames = [int(k)] + [int(k) - g for g in gaps]
            true = {f: traj.pose(f) for f in frames}
            for ci, (c, (pj, aj, pb, ab)) in enumerate(CONDITIONS.items()):
                # bias: one draw per flight and condition; jitter: one draw per frame
                rb = np.random.default_rng([seed, tnum, ci])
                bp = rb.normal(0, pb, 3)
                ba = rb.normal(0, np.radians(ab), 3)
                noisy = {}
                for f in frames:
                    rj = np.random.default_rng([seed, tnum, ci, f])
                    noisy[f] = perturb(
                        true[f], bp + rj.normal(0, pj, 3), ba + rj.normal(0, np.radians(aj), 3)
                    )
                sc = replace(s, sigma_pos_m=float(pj), sigma_att_rad=float(np.radians(aj)))
                P, var_z, _, uv_pts = points_from_tracks(
                    cam,
                    noisy[int(k)],
                    uvk,
                    tracks,
                    [(noisy[int(k) - g], sig_by_gap[g]) for g in gaps],
                    sc,
                )
                res = range_targets(cam, noisy[int(k)], uv, P, var_z, sc, uv_points=uv_pts)
                out[c]["terrain"][idx] = np.where(res["valid"], res["range_m"], np.nan)
                C, D = pixel_to_world_ray(cam, noisy[int(k)], uv)
                h = r["z_oracle"][idx] - C[2]
                lam = np.where((D[:, 2] > 0) & (h > 0), h / D[:, 2], np.nan)
                out[c]["flat"][idx] = lam * np.linalg.norm(D, axis=1)
    finally:
        traj.close()
    return out


def summarise(
    est: np.ndarray,
    truth: np.ndarray,
    land: np.ndarray,
    g: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    rel = np.abs(est - truth) / truth
    ok = land & np.isfinite(rel)
    a = rel[ok]
    return {
        "valid_fraction": float(ok.sum() / land.sum()),
        "median": float(np.median(a)),
        "median_ci": st.group_bootstrap_ci(a, g[ok], np.median, n_boot, rng),
        "p90": float(np.percentile(a, 90)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="*", default=None)
    ap.add_argument("--frame-step", type=int, default=1, help="use every n-th analysis frame")
    ap.add_argument("--check-exact", action="store_true", help="compare 'exact' with the cache")
    ap.add_argument("--workers", type=int, default=1, help="flights processed in parallel")
    args = ap.parse_args(argv)
    data = load_config("data")
    root = (
        REPO_ROOT
        / data.require("data_root")
        / data.require("midair_subdir")
        / data.require("midair_climate")
    )
    seed = load_config("project").require("seed")
    n_boot = load_config("uncertainty").require("n_boot")
    s = motion_settings()
    names = args.trajectories or [
        n
        for n in split_names("development")
        if any((root / "color_left" / n).glob("*.JPEG"))  # frames on disk, not archived
        and (CACHE / f"{n}.terrain.npz").exists()
    ]
    est = {c: {"terrain": [], "flat": []} for c in CONDITIONS}
    truth, land, groups, used = [], [], [], []
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outs = list(
            pool.map(
                run_trajectory,
                [root] * len(names),
                names,
                [s] * len(names),
                [args.frame_step] * len(names),
                [seed] * len(names),
            )
        )
    for name, o in zip(names, outs, strict=True):
        r = dict(np.load(CACHE / f"{name}.npz"))
        done = np.isin(r["frame"], np.unique(r["frame"])[:: args.frame_step])
        if args.check_exact:
            ref = dict(np.load(CACHE / f"{name}.terrain.npz"))
            a = np.where(ref["terrain.valid"], ref["terrain.range_m"], np.nan)[done]
            b = o["exact"]["terrain"][done]
            same = np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9)
            print(f"{name}: exact condition reproduces the saved results: {same}", flush=True)
        for c in CONDITIONS:
            for m in ("terrain", "flat"):
                est[c][m].append(o[c][m][done])
        truth.append(r["true_range_m"][done])
        land.append((r["cls"] != WATER_CLASS)[done])
        groups.append(np.full(done.sum(), int(name.split("_")[-1])))
        used.append(name)
        print(f"{name}: done", flush=True)
    truth, land, groups = map(np.concatenate, (truth, land, groups))
    rng = np.random.default_rng(seed)
    results = {}
    for c, spec in CONDITIONS.items():
        results[c] = {
            "pos_jitter_m": spec[0],
            "att_jitter_deg": spec[1],
            "pos_bias_m": spec[2],
            "att_bias_deg": spec[3],
            **{
                m: summarise(np.concatenate(est[c][m]), truth, land, groups, n_boot, rng)
                for m in ("terrain", "flat")
            },
        }
        t, f = results[c]["terrain"], results[c]["flat"]
        print(
            f"{c:20s} terrain {t['median']:.3f} (valid {t['valid_fraction']:.2f}, "
            f"p90 {t['p90']:.2f})  flat {f['median']:.3f} (p90 {f['p90']:.2f})"
        )
    rec = {
        "meta": {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "decision": "D-621",
            "split": "development",
            "trajectories": used,
            "frame_step": args.frame_step,
            "seed": seed,
            "n_boot": n_boot,
            "noise": "per-axis Gaussian sigma; jitter drawn per frame, bias once per flight",
            "nav_grades": "nav_good / nav_typical / nav_poor are illustrative assumptions",
            "metric": "land samples, |relative range error|; valid_fraction of land samples",
        },
        "conditions": results,
    }
    out = REPO_ROOT / "results/midair/development/terrain"
    (out / "pose_sweep.json").write_text(json.dumps(rec, indent=1) + "\n")
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for kind, key, what, unit, logx in (
        ("pos_jitter", "pos_jitter_m", "position", "m", True),
        ("att_jitter", "att_jitter_deg", "attitude", "deg", True),
    ):
        cs = [c for c in results if c.startswith(kind)]
        if not cs:
            continue
        xs = [results[c][key] for c in cs]
        line_chart(
            figdir / f"pose_sweep_{kind}.png",
            {
                "Terrain from motion": {
                    "x": xs,
                    "y": [results[c]["terrain"]["median"] for c in cs],
                },
                "Flat ground, true height": {
                    "x": xs,
                    "y": [results[c]["flat"]["median"] for c in cs],
                },
            },
            title=f"Range error against per-frame {what} noise",
            xlabel=f"Per-frame {what} noise, sigma per axis ({unit})",
            ylabel="Median |relative range error|",
            logx=logx,
            logy=True,
            note=f"Mid-Air development flights ({len(used)}), land samples; exact poses: "
            f"terrain {results['exact']['terrain']['median']:.1%}, "
            f"flat {results['exact']['flat']['median']:.1%}. D-621.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
