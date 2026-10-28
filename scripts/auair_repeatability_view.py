"""Repeatability of stationary vehicles on all AU-AIR views with per-frame tilt (D-513, D-515).

Usage: python scripts/auair_repeatability_view.py --source B2_yolo26n_960 --split calibration
       python scripts/auair_repeatability_view.py --source truth --split held_out --final

Every clip of the split (straight-down, oblique and mixed). Each frame of a stationary segment
(scripts/auair_stationary.py, run `--source`) gives a ground position: GPS position + the
ground offset of the box's contact point along its viewing ray (`ground_offset_oblique`), at
the drift-corrected height, the fitted focal length, the frame's tilt from motion
(scripts/auair_tilt.py; frames without a tilt are skipped) and the mount and camera model fixed
on the calibration clips (D-511: image up is forward, stabilised). Contact point: box
bottom-centre (primary; the object's ground contact in an oblique view) and box centre.
Spread = RMS distance about the segment mean, over segments with at least `MIN_FRAMES` usable
frames. Breakdowns: hover against all flight, height, tilt, image radius. The held-out split
runs once with --final. Output: results/auair/repeatability_view_<split>_<source>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aerialrange.evaluate.auair_world import ground_offset_oblique, spread_rms
from aerialrange.utils.config import REPO_ROOT, load_config

HOVER_SPEED_MS, HOVER_MIN_S = 0.12, 4.0  # D-511
MIN_FRAMES = 20
N_BOOT = 2000


def hover_mask(speed: np.ndarray, t: np.ndarray) -> np.ndarray:
    ok, out, i = speed < HOVER_SPEED_MS, np.zeros(len(speed), bool), 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        if t[j] - t[i] >= HOVER_MIN_S:
            out[i : j + 1] = True
        i = j + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="detector run name, or 'truth'")
    ap.add_argument("--split", required=True, choices=["calibration", "held_out"])
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "held_out" and not args.final:
        print("Held-out clips run once, after the settings are frozen. Use --final.")
        return 1
    au = load_config("auair")
    f = float(au.require("focal_px"))
    calib = set(au.require("calib_recordings"))
    drift = json.loads((REPO_ROOT / "results" / "auair" / "camera_scale.json").read_text())[
        "altimeter_drift"
    ]
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    rec_of = fr.groupby("clip")["recording"].first()
    clips = sorted(
        int(c) for c in rec_of.index if (rec_of[c] in calib) == (args.split == "calibration")
    )
    tracks_dir = REPO_ROOT / "cache" / "auair" / "runs" / args.source
    seg_dir = REPO_ROOT / "cache" / "auair" / "stationary" / args.source
    tilt_dir = REPO_ROOT / "cache" / "auair" / "tilt"
    rows = []
    for c in clips:
        sp, tp = seg_dir / f"clip_{c:02d}.csv", tilt_dir / f"clip_{c:02d}_frames.csv"
        if not sp.exists() or sp.stat().st_size < 5 or not tp.exists():
            continue
        g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
        rec = g["recording"][0]
        if rec in drift:
            d = drift[rec]
            h0 = (d["a_m"] + d["b_m_per_min"] / 60 * g["t_ms"] / 1000).to_numpy()
        else:
            h0 = np.zeros(len(g))
        t = g["t_ms"].to_numpy() / 1000
        h = g["alt_m"].to_numpy() + h0
        hov = hover_mask(np.hypot(g["linear_x"], g["linear_y"]).to_numpy(), t)
        tilt = pd.read_csv(tp)["tilt_deg"].to_numpy()
        tr = np.loadtxt(tracks_dir / f"clip_{c:02d}_vehicle.txt", delimiter=",", ndmin=2)
        for _, s in pd.read_csv(sp).iterrows():
            r = tr[
                (tr[:, 1] == s["track"])
                & (tr[:, 0] >= s["first_frame"])
                & (tr[:, 0] <= s["last_frame"])
            ]
            k = r[:, 0].astype(int) - 1
            keep = np.isfinite(tilt[k])
            if keep.sum() < MIN_FRAMES:
                continue
            r, k = r[keep], k[keep]
            u_c, v_c = r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5] / 2
            out = {
                "clip": c,
                "track": int(s["track"]),
                "frames": int(len(k)),
                "alt_m": float(g["alt_m"].to_numpy()[k].mean()),
                "tilt_deg": float(np.median(tilt[k])),
                "radius": float(np.median(np.hypot(u_c - 960, v_c - 540) / np.hypot(960, 540))),
            }
            for mode, v in (("bottom", r[:, 3] + r[:, 5]), ("centre", v_c)):
                dn, de = ground_offset_oblique(
                    u_c, v, h[k], f, g["angle_psi"].to_numpy()[k], tilt[k]
                )
                pts = np.column_stack([g["n_m"].to_numpy()[k] + dn, g["e_m"].to_numpy()[k] + de])
                ok = np.isfinite(pts).all(axis=1)
                if mode == "bottom":
                    # slant range and line-of-sight depression of the contact point (D-517)
                    out["range_m"] = float(np.nanmedian(np.hypot(np.hypot(dn, de), h[k])))
                    out["depression_deg"] = float(
                        np.nanmedian(tilt[k] + np.degrees(np.arctan((v - 540.0) / f)))
                    )
                for tag, m in (("all", ok), ("hover", ok & hov[k])):
                    sp = spread_rms(pts[m]) if m.sum() >= MIN_FRAMES else np.nan
                    out[f"{tag}_{mode}"] = sp
                    out[f"{tag}_{mode}_rel"] = sp / out["range_m"] if np.isfinite(sp) else np.nan
            rows.append(out)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(load_config("project").require("seed"))

    def summ(x: pd.Series) -> dict:
        x = x.dropna().to_numpy()
        if not len(x):
            return {"n": 0}
        b = [np.median(rng.choice(x, len(x))) for _ in range(N_BOOT)] if len(x) > 1 else [x[0]]
        return {
            "n": int(len(x)),
            "median_m": float(np.median(x)),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "p90_m": float(np.percentile(x, 90)),
        }

    def bands(col: str, edges: list[float], metric: str) -> dict:
        out = {}
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (df[col] >= lo) & (df[col] < hi)
            out[f"{lo}-{hi}"] = summ(df.loc[m, metric]) if len(df) else {"n": 0}
        return out

    res = {"n_segments": int(len(df))}
    if len(df):
        for mode in ("bottom", "centre"):
            res[mode] = {
                "hover": summ(df[f"hover_{mode}"]),
                "all_flight": summ(df[f"all_{mode}"]),
                "hover_relative": summ(df[f"hover_{mode}_rel"]),
                "all_flight_relative": summ(df[f"all_{mode}_rel"]),
            }
        dep = [0, 10, 20, 40, 60, 91]
        res["by_depression_all_bottom_m"] = bands("depression_deg", dep, "all_bottom")
        res["by_depression_all_bottom_relative"] = bands("depression_deg", dep, "all_bottom_rel")
        res["by_depression_hover_bottom_relative"] = bands(
            "depression_deg", dep, "hover_bottom_rel"
        )
        res["by_altitude_all_bottom_relative"] = bands("alt_m", [0, 15, 25, 40], "all_bottom_rel")
        res["by_tilt_all_bottom_relative"] = bands("tilt_deg", [0, 30, 60, 96], "all_bottom_rel")
        res["by_radius_all_bottom_relative"] = bands(
            "radius", [0, 0.3, 0.6, 1.01], "all_bottom_rel"
        )
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-517",
        "split": args.split,
        "source": args.source,
        "clips_with_segments": sorted(int(x) for x in df["clip"].unique()) if len(df) else [],
        "focal_px": f,
        "hover": {"speed_ms": HOVER_SPEED_MS, "min_s": HOVER_MIN_S},
        "min_frames": MIN_FRAMES,
        "results": res,
        "caveats": "No ground truth position: spread measures repeatability, not accuracy. "
        "Tilt from motion (about +/-5 deg, D-514); principal point at the image centre; flat "
        "ground; altimeter offset from climbs where available (recording 20190905142119 weak; "
        "held-out recordings without a calibration climb use the altimeter as recorded); "
        "the all-flight spread includes GPS drift; segments within a clip are correlated.",
    }
    dest = REPO_ROOT / "results" / "auair" / f"repeatability_view_{args.split}_{args.source}.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n")
    df.to_csv(
        REPO_ROOT / "cache" / "auair" / f"repeatability_view_{args.split}_{args.source}.csv",
        index=False,
    )
    print(json.dumps(out["results"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
