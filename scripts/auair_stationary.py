"""Find stationary vehicles in AU-AIR tracks from image motion only (D-510).

Usage: python scripts/auair_stationary.py --run B2_yolo26n_960 [--clips 0 1]
       python scripts/auair_stationary.py --run truth --noise-from B2_yolo26n_960 [--clips ...]

`--run truth` uses AU-AIR's own vehicle boxes (Car, Truck, Van, Bus) instead of detector
tracks (D-513): boxes are linked frame to frame by carrying each box through the measured
camera motion and matching it to the next frame's boxes at IoU >= `LINK_IOU` (greedy, highest
IoU first). The linked tracks are written to cache/auair/runs/truth/. Annotation noise is
unknown, so the detector's box noise (`--noise-from`) stands in for it (stated assumption).

Per clip: homographies between consecutive frames from background corners (all tracked boxes
masked), at half size; then residual steps of each vehicle track's box centre
(`evaluate/stationary.py`). Threshold (`{{STATIONARY_PX_THRESHOLD}}`), per track:
3 x sqrt(2) x sigma_pos, with sigma_pos^2 = (kappa x box size)^2 + sigma_H^2, where kappa is the
detector's vehicle box noise (max of kappa_u, kappa_v; results/visdrone/val/
pixel_noise_B2_yolo26n_960.json) and sigma_H the clip's homography noise (robust spread of the
RANSAC inlier residuals, full-size px). The start-to-end residual of a parked object is the
difference of two noisy positions, hence sqrt(2); 3 for a three-sigma bound.
`{{STATIONARY_MIN_S}}` = 4 s (20 frames at 5 fps; D-510). No altitude or GPS is read.
Output: cache/auair/stationary/<run>/clip_NN.csv (segments) and results/auair/stationary_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import cv2
import numpy as np
import pandas as pd

from aerialrange.evaluate.stationary import residual_steps, stationary_segments
from aerialrange.utils.config import REPO_ROOT, load_config

SC = 0.5
MIN_S = 4.0
LINK_IOU = 0.3
TRUTH_CLASSES = (1, 2, 3, 6)  # Car, Truck, Van, Bus (D-507)
MAX_GAP = 5  # bridge detection misses up to 1 s (5 frames) through chained camera motion
N_SIGMA = 3.0


def homography(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Full-size homography from frame a to b, and the robust spread (px) of its inlier
    residuals; identity and NaN if the fit fails."""
    p = cv2.goodFeaturesToTrack(a, 1500, 0.01, 6, mask=mask)
    if p is None or len(p) < 40:
        return np.eye(3), np.nan
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=4)
    r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21, 21), maxLevel=4)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((r - p)[:, 0], axis=1) < 1.0)
    if ok.sum() < 40:
        return np.eye(3), np.nan
    H, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 2.0)
    if H is None:
        return np.eye(3), np.nan
    S = np.diag([SC, SC, 1.0])
    Hf = np.linalg.inv(S) @ H @ S
    res = np.linalg.norm(
        cv2.perspectiveTransform(p[ok][inl[:, 0] == 1], H) - q[ok][inl[:, 0] == 1], axis=2
    )
    return Hf, float(1.4826 * np.median(res) / SC)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between ltwh boxes a (n, 4) and b (m, 4)."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1, None], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def warp_boxes(boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Carry ltwh boxes through homography H (axis-aligned box around the warped corners)."""
    if not len(boxes):
        return boxes
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    pts = np.stack([np.column_stack(c) for c in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))], axis=1)
    hom = np.concatenate([pts, np.ones(pts.shape[:2] + (1,))], axis=2) @ H.T
    xy = hom[..., :2] / hom[..., 2:]
    lo, hi = xy.min(axis=1), xy.max(axis=1)
    return np.column_stack([lo, hi - lo])


def link_truth(per_frame: list[np.ndarray], Hs: list[np.ndarray], max_gap: int = 5) -> np.ndarray:
    """Tracks (frame, id, l, t, w, h, 1.0) from per-frame boxes, linked through the camera
    motion by greedy IoU >= LINK_IOU. A track missing from the annotations for up to `max_gap`
    frames is carried forward through the chained camera motion and can still be matched."""
    rows, next_id = [], 1
    active: dict[int, list] = {}  # id -> [box carried into the current frame, frames unseen]
    for k, cur in enumerate(per_frame):
        if k > 0:
            for a in active.values():
                a[0] = warp_boxes(a[0][None, :], Hs[k - 1])[0]
                a[1] += 1
        ids_list = list(active)
        new_ids = np.zeros(len(cur), int)
        if ids_list and len(cur):
            iou = iou_matrix(np.array([active[i][0] for i in ids_list]), cur)
            used_a, used_b = set(), set()
            for flat in np.argsort(-iou, axis=None):
                i, j = np.unravel_index(flat, iou.shape)
                if iou[i, j] < LINK_IOU:
                    break
                if i in used_a or j in used_b:
                    continue
                used_a.add(i)
                used_b.add(j)
                new_ids[j] = ids_list[i]
        for j in range(len(cur)):
            if new_ids[j] == 0:
                new_ids[j], next_id = next_id, next_id + 1
            active[int(new_ids[j])] = [cur[j].copy(), 0]
            rows.append((k + 1, new_ids[j], *cur[j], 1.0))
        active = {i: a for i, a in active.items() if a[1] < max_gap}
    return np.array(rows, float).reshape(-1, 7)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    ap.add_argument("--noise-from", default=None, help="detector run whose box noise to use")
    args = ap.parse_args(argv)
    truth = args.run == "truth"
    data = load_config("data")
    root = REPO_ROOT / data.require("data_root") / data.require("auair_subdir")
    noise = json.loads(
        (
            REPO_ROOT
            / "results"
            / "visdrone"
            / "val"
            / f"pixel_noise_{args.noise_from or args.run}.json"
        ).read_text()
    )["groups"]["vehicle"]
    kappa = max(noise["kappa_u"], noise["kappa_v"])
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    runs = REPO_ROOT / "cache" / "auair" / "runs" / args.run
    out = REPO_ROOT / "cache" / "auair" / "stationary" / args.run
    out.mkdir(parents=True, exist_ok=True)
    clips = args.clips if args.clips else sorted(fr["clip"].unique())
    gt = {}
    if truth:
        runs.mkdir(parents=True, exist_ok=True)
        with (REPO_ROOT / "cache" / "auair" / "boxes.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                gt[r["image_name"]] = r["boxes"]
    summary = {}
    for c in clips:
        g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
        if truth:
            per_frame = [
                np.array(
                    [
                        [b["left"], b["top"], b["width"], b["height"]]
                        for b in gt[n]
                        if b["class"] in TRUTH_CLASSES
                    ],
                    float,
                ).reshape(-1, 4)
                for n in g["image_name"]
            ]
            mask_boxes = [
                np.array(
                    [[b["left"], b["top"], b["width"], b["height"]] for b in gt[n]], float
                ).reshape(-1, 4)
                for n in g["image_name"]
            ]
        else:
            tracks = np.loadtxt(runs / f"clip_{c:02d}_vehicle.txt", delimiter=",", ndmin=2)
            if not len(tracks):
                summary[int(c)] = {"tracks": 0, "segments": 0}
                continue
            mask_boxes = [tracks[tracks[:, 0] == k + 1][:, 2:6] for k in range(len(g))]
        prev, Hs, sig = None, [], []
        for k, p in enumerate(g["image_path"]):
            im = cv2.resize(cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE), None, fx=SC, fy=SC)
            if prev is not None:
                m = np.full(prev.shape, 255, np.uint8)
                for x, y, w, h in mask_boxes[k - 1]:  # boxes in the previous frame
                    x0, y0 = int(x * SC) - 4, int(y * SC) - 4
                    m[max(0, y0) : int((y + h) * SC) + 4, max(0, x0) : int((x + w) * SC) + 4] = 0
                H, s = homography(prev, im, m)
                Hs.append(H)
                sig.append(s)
            prev = im
        sigma_h = float(np.nanmedian(sig)) if np.isfinite(sig).any() else np.nan
        if truth:
            tracks = link_truth(per_frame, Hs)
            np.savetxt(runs / f"clip_{c:02d}_vehicle.txt", tracks, fmt="%.2f", delimiter=",")
            if not len(tracks):
                summary[int(c)] = {"tracks": 0, "segments": 0}
                continue
        t = g["t_ms"].to_numpy() / 1000
        rows = []
        for tid in np.unique(tracks[:, 1]):
            tr = tracks[tracks[:, 1] == tid]
            tr = tr[np.argsort(tr[:, 0])]
            cen = np.column_stack([tr[:, 0], tr[:, 2] + tr[:, 4] / 2, tr[:, 3] + tr[:, 5] / 2])
            size = float(np.median(np.maximum(tr[:, 4], tr[:, 5])))
            thr = N_SIGMA * np.sqrt(2) * np.hypot(kappa * size, sigma_h)
            steps = residual_steps(cen, Hs, MAX_GAP)
            tt = t[tr[:, 0].astype(int) - 1]
            for a, b in stationary_segments(steps, tt, thr, MIN_S):
                rows.append(
                    {
                        "track": int(tid),
                        "first_frame": int(tr[a, 0]),
                        "last_frame": int(tr[b, 0]),
                        "frames": b - a + 1,
                        "seconds": float(tt[b] - tt[a]),
                        "box_px": size,
                        "thr_px": float(thr),
                    }
                )
        pd.DataFrame(rows).to_csv(out / f"clip_{c:02d}.csv", index=False)
        summary[int(c)] = {
            "tracks": int(len(np.unique(tracks[:, 1]))),
            "segments": len(rows),
            "segment_frames": int(sum(r["frames"] for r in rows)),
            "homography_sigma_px": sigma_h,
            "homography_failures": int(np.isnan(sig).sum()),
        }
        print(f"clip {c}: {summary[int(c)]}", flush=True)
    dest = REPO_ROOT / "results" / "auair" / f"stationary_{args.run}.json"
    if dest.exists():  # merge: runs on different clip sets add to one summary
        prev = json.loads(dest.read_text())["clips"]
        summary = {**{int(k): v for k, v in prev.items()}, **summary}
    rec = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "decision": "D-510",
        "kappa_vehicle": kappa,
        "n_sigma": N_SIGMA,
        "min_s": MIN_S,
        "clips": dict(sorted(summary.items())),
    }
    dest.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
