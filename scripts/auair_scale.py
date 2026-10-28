"""AU-AIR focal length and altimeter drift from the calibration clips (D-508).

Usage: python scripts/auair_scale.py

Only straight-down clips (the manual labels, D-504) of the calibration recordings (D-506).
1. Altimeter offset h0 from the image scale change between frames 3, 5, 8 or 12 apart that
   differ by at least `MIN_CLIMB_M` in altitude; per recording, a robust linear drift in time.
2. Focal length from image-centre shift against GPS displacement between frames 5 or 10 apart
   with at least `MIN_MOVE_M` of motion, using the drift-corrected height.
3. Check: ground-truth car boxes (not touching the image border) give f x L from long side x
   corrected height; L = (f x L) / f.
Uncertainty: 95 % bootstrap intervals resampling whole clips (windows within a clip are
correlated). Frames are processed at half size; shifts are scaled back to full-size pixels.
Annotated objects are masked out before feature tracking (they move).
Output: results/auair/camera_scale.json (and cache/auair/scale_windows.csv).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import cv2
import numpy as np
import pandas as pd

from aerialrange.evaluate.auair_scale import (
    fit_linear_drift,
    focal_from_motion,
    h0_from_scale,
    similarity_motion,
)
from aerialrange.utils.config import REPO_ROOT, load_config

SC = 0.5
# Window thresholds (D-508): a 1.5 m climb at 20 m changes scale by about 7 %, and 2 m of
# motion moves the image about 150 px; both are far above the frame-to-frame measurement noise.
MIN_CLIMB_M = 1.5
MIN_MOVE_M = 2.0
# GPS displacement must agree with the displacement integrated from the recorded velocities
# (linear_x, linear_y) within this ratio; it rejects GPS jumps while hovering (D-508).
GPS_VEL_RATIO = 1.25
N_BOOT = 2000


def main() -> int:
    data, au = load_config("data"), load_config("auair")
    root = REPO_ROOT / data.require("data_root") / data.require("auair_subdir")
    seed = load_config("project").require("seed")
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    lab = pd.read_csv(REPO_ROOT / "results" / "auair" / "clip_labels.csv")
    calib = set(au.require("calib_recordings"))
    down = set(lab.loc[lab["view"] == "down", "clip"])
    clips = sorted(c for c in down if fr.loc[fr["clip"] == c, "recording"].iloc[0] in calib)
    boxes = {}
    with (REPO_ROOT / "cache" / "auair" / "boxes.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            boxes[r["image_name"]] = r["boxes"]

    def gray(name: str) -> np.ndarray:
        im = cv2.imread(str(root / "images" / name), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(im, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)

    def mask(name: str) -> np.ndarray:
        m = np.full((int(1080 * SC), int(1920 * SC)), 255, np.uint8)
        for b in boxes[name]:
            x0, y0 = int(b["left"] * SC), int(b["top"] * SC)
            x1, y1 = x0 + int(b["width"] * SC), y0 + int(b["height"] * SC)
            m[max(0, y0 - 4) : y1 + 4, max(0, x0 - 4) : x1 + 4] = 0
        return m

    rows = []
    for c in clips:
        g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
        alt, t = g["alt_m"].to_numpy(), g["t_ms"].to_numpy() / 1000
        n, e = g["n_m"].to_numpy(), g["e_m"].to_numpy()
        vx, vy = g["linear_x"].to_numpy(), g["linear_y"].to_numpy()
        cache: dict[int, np.ndarray] = {}

        def img(k: int, g: pd.DataFrame = g, cache: dict = cache) -> np.ndarray:
            if k not in cache:
                if len(cache) > 30:
                    cache.clear()
                cache[k] = gray(g["image_name"][k])
            return cache[k]

        for i in range(0, len(g), 2):
            for w in (3, 5, 8, 10, 12):
                j = i + w
                if j >= len(g):
                    continue
                climb = abs(alt[j] - alt[i]) >= MIN_CLIMB_M and w in (3, 5, 8, 12)
                d = float(np.hypot(n[j] - n[i], e[j] - e[i]))
                dtk = np.diff(t[i : j + 1])
                dv = float(
                    np.hypot(
                        np.sum(0.5 * (vx[i:j] + vx[i + 1 : j + 1]) * dtk),
                        np.sum(0.5 * (vy[i:j] + vy[i + 1 : j + 1]) * dtk),
                    )
                )
                agree = dv > 0 and 1 / GPS_VEL_RATIO <= d / dv <= GPS_VEL_RATIO
                move = d >= MIN_MOVE_M and w in (5, 10) and agree
                if not (climb or move):
                    continue
                res = similarity_motion(img(i), img(j), mask(g["image_name"][i]))
                if res is None:
                    continue
                shift, s, inl = res
                rows.append(
                    {
                        "clip": c,
                        "recording": g["recording"][0],
                        "i": i,
                        "w": w,
                        "t_s": (t[i] + t[j]) / 2,
                        "alt_i": alt[i],
                        "alt_j": alt[j],
                        "d_m": d,
                        "d_vel_m": dv,
                        "shift_px": shift / SC,
                        "scale": s,
                        "inliers": inl,
                        "kind": "climb" if climb else "move",
                        "h0": h0_from_scale(alt[i], alt[j], s) if climb else np.nan,
                    }
                )
        print(f"clip {c}: {sum(r['clip'] == c for r in rows)} windows", flush=True)
    win = pd.DataFrame(rows)
    win.to_csv(REPO_ROOT / "cache" / "auair" / "scale_windows.csv", index=False)

    drift = {}
    for rec, gw in win[win["kind"] == "climb"].groupby("recording"):
        # A drift slope needs climbs spread over time (at least two clips); otherwise a
        # constant offset (a short span cannot pin a slope: D-508).
        if gw["clip"].nunique() >= 2:
            a, b = fit_linear_drift(gw["t_s"], gw["h0"])
        else:
            a, b = float(gw["h0"].median()), 0.0
        res = gw["h0"] - (a + b * gw["t_s"])
        drift[rec] = {
            "a_m": a,
            "b_m_per_min": b * 60,
            "n_windows": len(gw),
            "clips": sorted(int(x) for x in gw["clip"].unique()),
            "model": "linear drift" if gw["clip"].nunique() >= 2 else "constant (one clip)",
            "residual_robust_sigma_m": float(1.4826 * np.median(np.abs(res - res.median()))),
        }
    mv = win[win["kind"] == "move"].copy()
    has = mv["recording"].isin(drift)
    mv = mv[has]
    h0_t = np.array(
        [
            drift[r]["a_m"] + drift[r]["b_m_per_min"] / 60 * ts
            for r, ts in zip(mv["recording"], mv["t_s"], strict=True)
        ]
    )
    mv["h_true"] = (mv["alt_i"] + mv["alt_j"]) / 2 + h0_t
    mv["f"] = [
        focal_from_motion(s, d, h)
        for s, d, h in zip(mv["shift_px"], mv["d_m"], mv["h_true"], strict=True)
    ]

    # Car check: long side of ground-truth car boxes x corrected height.
    cars = []
    for c in clips:
        g = fr[fr["clip"] == c]
        rec = g["recording"].iloc[0]
        if rec not in drift:
            continue
        for _, r in g.iterrows():
            h = r["alt_m"] + drift[rec]["a_m"] + drift[rec]["b_m_per_min"] / 60 * r["t_ms"] / 1000
            for b in boxes[r["image_name"]]:
                if b["class"] != 1:
                    continue
                x0, y0, bw, bh = b["left"], b["top"], b["width"], b["height"]
                if x0 <= 1 or y0 <= 1 or x0 + bw >= 1919 or y0 + bh >= 1079:
                    continue
                cars.append({"clip": c, "fL": max(bw, bh) * h})
    cars = pd.DataFrame(cars)

    rng = np.random.default_rng(seed)
    cl = sorted(mv["clip"].unique())
    by_clip = {k: v["f"].to_numpy() for k, v in mv.groupby("clip")}
    car_by = {k: v["fL"].to_numpy() for k, v in cars.groupby("clip")}
    f_boot, l_boot = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(cl, len(cl))
        f = np.median(np.concatenate([by_clip[k] for k in pick]))
        f_boot.append(f)
        cp = [k for k in rng.choice(sorted(car_by), len(car_by))]
        l_boot.append(np.median(np.concatenate([car_by[k] for k in cp])) / f)
    f_med = float(mv["f"].median())
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-508",
        "calibration_recordings": sorted(calib),
        "straight_down_clips_used": [int(c) for c in clips],
        "thresholds": {
            "min_climb_m": MIN_CLIMB_M,
            "min_move_m": MIN_MOVE_M,
            "gps_velocity_ratio": GPS_VEL_RATIO,
            "half_size": SC,
        },
        "altimeter_drift": drift,
        "focal_px": {
            "median": f_med,
            "ci95_clip_bootstrap": [
                float(np.percentile(f_boot, 2.5)),
                float(np.percentile(f_boot, 97.5)),
            ],
            "n_windows": int(len(mv)),
            "by_clip_median": {int(k): float(np.median(v)) for k, v in by_clip.items()},
            "horizontal_fov_deg": float(np.degrees(2 * np.arctan(960 / f_med))),
        },
        "car_check": {
            "n_boxes": int(len(cars)),
            "f_times_L_median_px_m": float(cars["fL"].median()),
            "implied_car_length_m": float(cars["fL"].median() / f_med),
            "ci95_clip_bootstrap": [
                float(np.percentile(l_boot, 2.5)),
                float(np.percentile(l_boot, 97.5)),
            ],
            "note": "Axis-aligned boxes: the long side of a car at an angle overstates its length, "
            "so this is an upper-leaning estimate. Compare with a published mean car length.",
        },
        "n_boot": N_BOOT,
    }
    dest = REPO_ROOT / "results" / "auair" / "camera_scale.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("altimeter_drift", "focal_px", "car_check")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
