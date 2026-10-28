"""Moving targets on AU-AIR: ranging plus the ground-plane Kalman filter (D-518).

Usage: python scripts/auair_moving.py --run B2_yolo26n_960 --tune      (calibration only)
       python scripts/auair_moving.py --run B2_yolo26n_960             (all clips)

Each tracked object's box bottom-centre is placed on the ground in every frame that has a
camera tilt (scripts/auair_tilt.py), as in the repeatability analysis (ground_offset_oblique,
drift-corrected height, focal 1,259 px, D-511 mount). Measurement noise is measured, not
assumed: sigma = (frame-to-frame jitter of parked objects' ground points on the calibration
clips, relative to slant range, for the measurement's line-of-sight depression band) x slant
range (`jitter_sigma`, D-518). The constant-velocity ground filter
(geometry/kalman.py) runs in world coordinates, predicting through frames without
a measurement. `{{KALMAN_Q}}` is chosen on the calibration recordings only, as the value whose
median normalised innovation squared (NIS, 2 degrees of freedom) is closest to its expected
median 2 ln 2 = 1.386 over a grid (the median resists the rare huge innovations of identity
switches, which dominate the mean).
Speeds are a plausibility check only; AU-AIR has no ground-truth speeds, so no accuracy claim
is made. Output: results/auair/moving_<run>.json, results/auair/figures/speed_<run>.png (+ CSV),
and cache/auair/tracks_with_range_<run>.csv.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from aerialrange.evaluate.auair_world import ground_offset_oblique
from aerialrange.geometry.kalman import GroundKalman
from aerialrange.utils.config import REPO_ROOT, load_config

Q_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10.0, 30.0)
NIS_MEDIAN_2DOF = 1.3863  # median of chi-squared with 2 degrees of freedom: 2 ln 2
MIN_MEAS = 10
FRAME_S = 0.2  # 5 fps (configs/auair.yaml frame_interval_ms)


EDGES = np.array([(0, 10), (10, 20), (20, 40), (40, 60), (60, 91)], float)


def jitter_sigma(run: str, au: dict, drift: dict, fr: pd.DataFrame, clips: list[int]) -> np.ndarray:
    """Frame-to-frame measurement noise of ground points, relative to slant range, per
    line-of-sight depression band, measured on stationary segments of the given clips: the
    robust spread of consecutive differences of a parked object's ground point, divided by
    sqrt(2) (each difference holds two noisy positions). Slow drift (GPS, altimeter) is not
    frame-to-frame noise and is excluded by differencing (D-518)."""
    vals: list[list[float]] = [[] for _ in EDGES]
    seg_dir = REPO_ROOT / "cache" / "auair" / "stationary" / run
    for c in clips:
        sp = seg_dir / f"clip_{c:02d}.csv"
        if not sp.exists() or sp.stat().st_size < 5:
            continue
        segs = pd.read_csv(sp)
        ms = measurements(c, run, au, drift, fr, EDGES, np.ones(len(EDGES)), raw=True)
        for _, s in segs.iterrows():
            for tid, fs, pts, slant, dep in ms.get("vehicle", []):
                if tid != s["track"]:
                    continue
                m = (fs >= s["first_frame"]) & (fs <= s["last_frame"])
                f2, p2 = fs[m], pts[m]
                cons = np.flatnonzero(np.diff(f2) == 1)
                if len(cons) < 5:
                    continue
                dpt = p2[cons + 1] - p2[cons]
                rob = 1.4826 * np.median(np.abs(dpt - np.median(dpt, axis=0)), axis=0)
                sig = float(np.hypot(*rob) / np.sqrt(2) / np.sqrt(2))  # per axis, then per point
                b = int(
                    np.clip(
                        np.searchsorted(EDGES[:, 1], np.median(dep[m]), side="right"),
                        0,
                        len(EDGES) - 1,
                    )
                )
                vals[b].append(sig / float(np.median(slant[m])))
    rel = np.array([np.median(v) if v else np.nan for v in vals])
    good = np.flatnonzero(np.isfinite(rel))
    for i in range(len(rel)):
        if not np.isfinite(rel[i]):
            rel[i] = rel[good[np.argmin(np.abs(good - i))]]
    return rel


def measurements(
    c: int,
    run: str,
    au: dict,
    drift: dict,
    fr: pd.DataFrame,
    edges: np.ndarray,
    rel: np.ndarray,
    raw: bool = False,
) -> dict:
    """Per group, per track: frames, ground points (n, e) and sigma (m)."""
    f = float(au["focal_px"])
    g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
    tp = REPO_ROOT / "cache" / "auair" / "tilt" / f"clip_{c:02d}_frames.csv"
    if not tp.exists():
        return {}
    tilt = pd.read_csv(tp)["tilt_deg"].to_numpy()
    rec = g["recording"][0]
    d = drift.get(rec)
    h0 = (d["a_m"] + d["b_m_per_min"] / 60 * g["t_ms"] / 1000).to_numpy() if d else np.zeros(len(g))
    h = g["alt_m"].to_numpy() + h0
    out = {}
    for grp in ("vehicle", "person"):
        p = REPO_ROOT / "cache" / "auair" / "runs" / run / f"clip_{c:02d}_{grp}.txt"
        if not p.exists() or p.stat().st_size == 0:
            continue
        tr = np.loadtxt(p, delimiter=",", ndmin=2)
        for tid in np.unique(tr[:, 1]):
            r = tr[tr[:, 1] == tid]
            r = r[np.argsort(r[:, 0])]
            k = r[:, 0].astype(int) - 1
            ok = np.isfinite(tilt[k])
            if ok.sum() < MIN_MEAS:
                continue
            u, v = r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5]
            dn, de = ground_offset_oblique(u, v, h[k], f, g["angle_psi"].to_numpy()[k], tilt[k])
            pts = np.column_stack([g["n_m"].to_numpy()[k] + dn, g["e_m"].to_numpy()[k] + de])
            dep = tilt[k] + np.degrees(np.arctan((v - 540.0) / f))
            slant = np.hypot(np.hypot(dn, de), h[k])
            band = np.clip(np.searchsorted(edges[:, 1], dep, side="right"), 0, len(rel) - 1)
            sig = rel[band] * slant
            m = ok & np.isfinite(pts).all(axis=1) & np.isfinite(sig)
            if m.sum() >= MIN_MEAS:
                extra = (slant[m], dep[m]) if raw else (sig[m],)
                out.setdefault(grp, []).append((int(tid), r[m, 0].astype(int), pts[m], *extra))
    return out


def filter_track(
    frames: np.ndarray, pts: np.ndarray, sig: np.ndarray, q: float
) -> tuple[list, list]:
    """Run the filter over the track's frame span (predict-only where unmeasured). Returns
    per-measurement NIS values and per-frame (frame, speed, speed sigma, n, e)."""
    R = [np.eye(2) * s**2 for s in sig]
    kf = GroundKalman.from_two_points(
        pts[0], R[0], pts[1], R[1], (frames[1] - frames[0]) * FRAME_S, q
    )
    nis, rows = [], []
    last = frames[1]
    for j in range(2, len(frames)):
        kf.predict((frames[j] - last) * FRAME_S)
        last = frames[j]
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = pts[j] - Hm @ kf.x
        S = Hm @ kf.P @ Hm.T + R[j]
        nis.append(float(y @ np.linalg.solve(S, y)))
        kf.update(pts[j], R[j])
        sp, sp_sig, _, _ = kf.speed_heading()
        rows.append((frames[j], sp, sp_sig, *kf.position))
    return nis, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tune", action="store_true", help="choose KALMAN_Q on calibration clips")
    args = ap.parse_args(argv)
    au_cfg = load_config("auair")
    au = {"focal_px": au_cfg.require("focal_px")}
    calib = set(au_cfg.require("calib_recordings"))
    drift = json.loads((REPO_ROOT / "results/auair/camera_scale.json").read_text())[
        "altimeter_drift"
    ]
    fr = pd.read_csv(REPO_ROOT / "cache/auair/frames.csv", dtype={"recording": str})
    rec_of = fr.groupby("clip")["recording"].first()
    cal_clips = [int(c) for c in rec_of.index if rec_of[c] in calib]
    edges, rel = EDGES, jitter_sigma(args.run, au, drift, fr, cal_clips)
    print(
        "relative frame-to-frame sigma by depression band:", np.round(rel, 4).tolist(), flush=True
    )
    if args.tune:
        meas = {c: measurements(c, args.run, au, drift, fr, edges, rel) for c in cal_clips}
        table = {}
        for q in Q_GRID:
            nis = [
                x
                for c in meas
                for grp in meas[c]
                for _, fs, pts, sg in meas[c][grp]
                for x in filter_track(fs, pts, sg, q)[0]
            ]
            table[q] = {
                "mean_nis": float(np.mean(nis)),
                "median_nis": float(np.median(nis)),
                "n": len(nis),
            }
            print(f"q {q}: mean NIS {table[q]['mean_nis']:.2f} (n {len(nis)})", flush=True)
        # median NIS, robust to the rare large jumps of identity switches (D-518)
        best = min(Q_GRID, key=lambda q: abs(table[q]["median_nis"] - NIS_MEDIAN_2DOF))
        out = {
            "decision": "D-518",
            "grid": table,
            "selected_q": best,
            "target_median_nis": NIS_MEDIAN_2DOF,
            "relative_sigma_by_depression": {
                f"{a:g}-{b:g}": float(v) for (a, b), v in zip(edges, rel, strict=True)
            },
        }
        (REPO_ROOT / "results/auair/kalman_q_tuning.json").write_text(
            json.dumps(out, indent=1) + "\n"
        )
        print(f"selected q = {best}")
        return 0
    q = float(load_config("geometry").require("kalman_q"))
    rows, per_track = [], []
    for c in sorted(int(x) for x in rec_of.index):
        split = "calibration" if rec_of[c] in calib else "held_out"
        for grp, tracks in measurements(c, args.run, au, drift, fr, edges, rel).items():
            for tid, fs, pts, sg in tracks:
                _, fr_rows = filter_track(fs, pts, sg, q)
                sp = np.array([x[1] for x in fr_rows[3:]])  # skip the first second of settling
                if not len(sp):
                    continue
                per_track.append(
                    {
                        "clip": c,
                        "split": split,
                        "group": grp,
                        "track": tid,
                        "measurements": len(fs),
                        "speed_median_ms": float(np.median(sp)),
                        "speed_p90_ms": float(np.percentile(sp, 90)),
                    }
                )
                rows += [(c, grp, tid, *x) for x in fr_rows]
    pt = pd.DataFrame(per_track)
    pd.DataFrame(
        rows,
        columns=["clip", "group", "track", "frame", "speed_ms", "speed_sigma_ms", "n_m", "e_m"],
    ).to_csv(REPO_ROOT / f"cache/auair/tracks_with_range_{args.run}.csv", index=False)
    summary = {}
    for grp, gg in pt.groupby("group"):
        kmh = gg["speed_median_ms"].to_numpy() * 3.6
        summary[grp] = {
            "tracks": int(len(gg)),
            "median_kmh": float(np.median(kmh)),
            "p90_kmh": float(np.percentile(kmh, 90)),
            "share_above_70_kmh": float((kmh > 70).mean()),
            "share_below_2_kmh": float((kmh < 2).mean()),
        }
    from aerialrange.viz.plots import line_chart

    bins = np.arange(0, 80, 5)
    series = {}
    for grp, gg in pt.groupby("group"):
        hist, _ = np.histogram(
            np.clip(gg["speed_median_ms"].to_numpy() * 3.6, 0, 79.9), bins=np.append(bins, 80)
        )
        series[grp] = {"x": list(bins + 2.5), "y": list(hist / hist.sum())}
    line_chart(
        REPO_ROOT / f"results/auair/figures/speed_{args.run}.png",
        series,
        "AU-AIR track speeds (plausibility only)",
        "Median speed of a track (km/h, 5 km/h bins)",
        "Share of tracks",
        note="No ground-truth speeds exist in AU-AIR; this checks plausibility, not accuracy. "
        "Tracks with at least 10 ranged frames; speeds above 80 km/h counted in the last bin.",
    )
    out = {
        "decision": "D-518",
        "run": args.run,
        "kalman_q": q,
        "summary": summary,
        "note": "Plausibility only: no ground-truth speeds.",
    }
    (REPO_ROOT / f"results/auair/moving_{args.run}.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
