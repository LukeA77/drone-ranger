"""Per-frame camera tilt on AU-AIR from ground motion (D-505, D-514).

Usage: python scripts/auair_tilt.py [--clips 28 31] [--workers 4]

Per clip: pairs of frames 1 and 2 s apart (every 3rd frame) whose camera moved at least
`MIN_MOTION_M` (GPS horizontal and altitude change combined); the ground-plane homography is
fitted to corners in the central 60 % of the image (lens distortion, D-514) with annotated
objects masked, and decomposed (`evaluate/tilt_motion.py`), keeping the solution whose rotation
matches the recorded yaw change. A frame's tilt is the median of the estimates whose window
midpoint lies within `NEIGH_S` seconds, if there are at least `MIN_EST`; otherwise NaN (the
camera's tilt cannot be seen while the drone hovers, so hover frames take it from nearby motion,
assuming the gimbal holds its angle). Output: cache/auair/tilt/clip_NN_windows.csv and
clip_NN_frames.csv (frame index, tilt, number of estimates used, their spread), and
results/auair/tilt_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aerialrange.utils.config import REPO_ROOT, load_config

SC = 0.5
CENTRAL = 0.6
MIN_MOTION_M = 1.5
NEIGH_S = 10.0
MIN_EST = 3


def run_clip(c: int) -> dict:
    import cv2

    from aerialrange.evaluate.tilt_motion import tilt_from_homography

    cv2.setNumThreads(2)
    data, au = load_config("data"), load_config("auair")
    root = REPO_ROOT / data.require("data_root") / data.require("auair_subdir")
    f = float(au.require("focal_px"))
    K = np.array([[f, 0, 960.0], [0, f, 540.0], [0, 0, 1.0]])
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
    boxes = {}
    with (REPO_ROOT / "cache" / "auair" / "boxes.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["image_name"] in set(g["image_name"]):
                boxes[r["image_name"]] = r["boxes"]

    def gray(n: str) -> np.ndarray:
        im = cv2.imread(str(root / "images" / n), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(im, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)

    def mask(n: str) -> np.ndarray:
        h, w = int(1080 * SC), int(1920 * SC)
        m = np.zeros((h, w), np.uint8)
        mh, mw = int(h * (1 - CENTRAL) / 2), int(w * (1 - CENTRAL) / 2)
        m[mh : h - mh, mw : w - mw] = 255
        for b in boxes.get(n, []):
            x0, y0 = int(b["left"] * SC), int(b["top"] * SC)
            m[
                max(0, y0 - 4) : y0 + int(b["height"] * SC) + 4,
                max(0, x0 - 4) : x0 + int(b["width"] * SC) + 4,
            ] = 0
        return m

    t = g["t_ms"].to_numpy() / 1000
    n, e, alt = g["n_m"].to_numpy(), g["e_m"].to_numpy(), g["alt_m"].to_numpy()
    psi = g["angle_psi"].to_numpy()
    rows = []
    for i in range(0, len(g) - 5, 3):
        for w in (5, 10):
            j = i + w
            if (
                j >= len(g)
                or np.hypot(np.hypot(n[j] - n[i], e[j] - e[i]), alt[j] - alt[i]) < MIN_MOTION_M
            ):
                continue
            a, b = gray(g["image_name"][i]), gray(g["image_name"][j])
            p = cv2.goodFeaturesToTrack(a, 2000, 0.01, 6, mask=mask(g["image_name"][i]))
            if p is None or len(p) < 60:
                continue
            q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(31, 31), maxLevel=5)
            r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(31, 31), maxLevel=5)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((r - p)[:, 0], axis=1) < 1.0)
            if ok.sum() < 60:
                continue
            H, inl = cv2.findHomography(p[ok] / SC, q[ok] / SC, cv2.RANSAC, 3.0)
            if H is None or inl.sum() < 60:
                continue
            dyaw = float(np.degrees(abs(np.angle(np.exp(1j * (psi[j] - psi[i]))))))
            tilt = tilt_from_homography(H, K, expected_rot_deg=dyaw, min_motion=0.03)
            if tilt is not None:
                rows.append(
                    {
                        "i": i,
                        "j": j,
                        "t_mid": (t[i] + t[j]) / 2,
                        "tilt": tilt,
                        "inliers": int(inl.sum()),
                    }
                )
    win = pd.DataFrame(rows, columns=["i", "j", "t_mid", "tilt", "inliers"])
    out = REPO_ROOT / "cache" / "auair" / "tilt"
    out.mkdir(parents=True, exist_ok=True)
    win.to_csv(out / f"clip_{c:02d}_windows.csv", index=False)
    tilt_f, n_f, sp_f = [], [], []
    for tk in t:
        near = win["tilt"][(win["t_mid"] - tk).abs() <= NEIGH_S].to_numpy()
        ok = len(near) >= MIN_EST
        tilt_f.append(float(np.median(near)) if ok else np.nan)
        n_f.append(len(near))
        sp_f.append(float(1.4826 * np.median(np.abs(near - np.median(near)))) if ok else np.nan)
    pd.DataFrame(
        {"frame": np.arange(1, len(g) + 1), "tilt_deg": tilt_f, "n_est": n_f, "spread_deg": sp_f}
    ).to_csv(out / f"clip_{c:02d}_frames.csv", index=False)
    tf = np.array(tilt_f)
    return {
        "clip": int(c),
        "frames": int(len(g)),
        "windows": int(len(win)),
        "frames_with_tilt": int(np.isfinite(tf).sum()),
        "tilt_median": float(np.nanmedian(tf)) if np.isfinite(tf).any() else None,
        "tilt_p10_p90": [float(np.nanpercentile(tf, 10)), float(np.nanpercentile(tf, 90))]
        if np.isfinite(tf).any()
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    clips = args.clips if args.clips else sorted(int(c) for c in fr["clip"].unique())
    sizes = fr.groupby("clip").size()
    clips = sorted(clips, key=lambda c: -sizes[c])  # longest first, for load balance
    dest = REPO_ROOT / "results" / "auair" / "tilt_summary.json"
    summary = json.loads(dest.read_text())["clips"] if dest.exists() else {}
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_clip, clips):
            summary[str(r["clip"])] = r
            print(r, flush=True)
    rec = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-514",
        "settings": {
            "central": CENTRAL,
            "min_motion_m": MIN_MOTION_M,
            "neigh_s": NEIGH_S,
            "min_est": MIN_EST,
        },
        "clips": dict(sorted(summary.items(), key=lambda kv: int(kv[0]))),
    }
    dest.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
