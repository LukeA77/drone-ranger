"""Measure feature-tracking noise against ground truth on the Mid-Air development split.

Usage: python scripts/measure_tracking_noise.py [--frame-step 100] [--gaps 1 2 5 10]

For corners detected in frame k (same detector and tracker settings as the terrain method),
the true 3D point (from radial depth) is projected into frame k - gap and compared with the
tracked position. Only corners on locally smooth depth (neighbour depths within 2 %) are used,
so the truth itself is not ambiguous at a depth edge. Reports a robust sigma per axis (1.4826
MAD), the median and 90th percentile of the error norm, and the bias, per gap. Development
split only. Writes results/midair/development/tracking_noise.json, the source
for the terrain method's `sigma_px` (`{{TERRAIN_SIGMA_PX}}`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import cv2
import numpy as np

from aerialrange.data.midair import (
    MidAirTrajectory,
    midair_camera,
    read_depth,
    sky_mask,
    world_points,
)
from aerialrange.evaluate.terrain_motion import detect_features, track_to
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import split_names  # noqa: E402
from midair_terrain import motion_settings  # noqa: E402

SMOOTH_REL = 0.02  # neighbour depths within 2 %: the truth at the pixel is unambiguous


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-step", type=int, default=100)
    ap.add_argument("--gaps", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    data = load_config("data")
    root = (
        REPO_ROOT
        / data.require("data_root")
        / data.require("midair_subdir")
        / data.require("midair_climate")
    )
    s = motion_settings(require_sigma=False)
    gaps = args.gaps or list(s.frame_gaps)
    names = [
        n
        for n in split_names("development")
        if (root / "color_left" / n / ".frames.done").exists()
        and not (root / "depth" / n / ".archived").exists()
    ]
    errs: dict[int, list[np.ndarray]] = {g: [] for g in gaps}
    for name in names:
        traj = MidAirTrajectory(root, name)
        cam = None
        for k in range(max(gaps), traj.n_frames, args.frame_step):
            img = cv2.imread(str(traj.frame_path("color_left", k)), cv2.IMREAD_GRAYSCALE)
            cam = cam or midair_camera(img.shape[1], img.shape[0])
            depth = read_depth(traj.frame_path("depth", k))
            p0 = detect_features(img, s)
            uvk = p0[:, 0].astype(float)
            h, w = depth.shape
            ui = np.clip(np.round(uvk[:, 0]).astype(int), 1, w - 2)
            vi = np.clip(np.round(uvk[:, 1]).astype(int), 1, h - 2)
            d = depth[vi, ui].astype(float)
            nb = np.stack([depth[vi + a, ui + b] for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0))])
            smooth = np.all(np.abs(nb - d) / d < SMOOTH_REL, axis=0) & ~sky_mask(d)
            Xt = world_points(cam, traj.pose(k), uvk, d, "radial")
            for g in gaps:
                img_p = cv2.imread(str(traj.frame_path("color_left", k - g)), cv2.IMREAD_GRAYSCALE)
                uvp, ok = track_to(img, img_p, p0, s)
                pose = traj.pose(k - g)
                Xc = (Xt - pose.p_w) @ (pose.R_wb @ cam.R_bc)
                front = Xc[:, 2] > 0
                proj = np.column_stack(
                    [cam.fx * Xc[:, 0] / Xc[:, 2] + cam.cx, cam.fy * Xc[:, 1] / Xc[:, 2] + cam.cy]
                )
                m = ok & smooth & front
                errs[g].append(uvp[m] - proj[m])
        traj.close()
        print(f"{name} done", flush=True)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "development",
        "trajectories": names,
        "frame_step": args.frame_step,
        "smooth_rel": SMOOTH_REL,
        "by_gap": {},
    }
    for g, lst in errs.items():
        e = np.concatenate(lst) if lst else np.zeros((0, 2))
        if not len(e):
            continue
        n = np.linalg.norm(e, axis=1)
        mad = 1.4826 * np.median(np.abs(e - np.median(e, axis=0)), axis=0)
        out["by_gap"][str(g)] = {
            "n": int(len(e)),
            "robust_sigma_u_px": float(mad[0]),
            "robust_sigma_v_px": float(mad[1]),
            "median_norm_px": float(np.median(n)),
            "p90_norm_px": float(np.percentile(n, 90)),
            "bias_px": [float(np.median(e[:, 0])), float(np.median(e[:, 1]))],
        }
    dest = REPO_ROOT / "results" / "midair" / "development" / "tracking_noise.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["by_gap"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
