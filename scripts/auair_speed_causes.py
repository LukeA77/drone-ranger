"""Why do some AU-AIR vehicle tracks read implausibly fast? Causes separated one at a time.

Usage: python scripts/auair_speed_causes.py --run B2_yolo26n_960 [--truth-run truth]

Version 1 (scripts/auair_moving.py, D-518) found that 11 % of AU-AIR vehicle tracks have a
median speed above 70 km/h. AU-AIR has no ground-truth speeds, so each candidate cause is tested
by recomputing the speeds with that cause removed and everything else as in version 1 (same
ground points, measurement noise and `kalman_q`); the share above 70 km/h is reported for each.

Candidate causes and how each is removed:
- frame timing: version 1 takes the time between two measurements as frame count x 0.2 s,
  but 4 % of AU-AIR frame steps are dropped frames (0.4 to 2 s). Removed by taking the
  time from the recorded timestamps.
- identity switches: the detector track is matched frame by frame to AU-AIR's own boxes,
  linked into tracks (`--truth-run`, scripts/auair_stationary.py); a change of the matched
  box track along a detector track is a switch. Removed by cutting tracks at switches.
  The same speeds computed on the linked annotation tracks show what is left without the
  detector and tracker at all.
- tilt changes: camera tilt is estimated per frame from ground motion (scripts/auair_tilt.py);
  a change of estimate within a track moves every ground point. Removed by holding the tilt at
  the track's median.
- shallow viewing angles: ground-position error grows as the line of sight flattens. Shown
  by the share above 70 km/h per depression band of the track's median line of sight; no
  correction is applied, since a flat line of sight is a property of the view, not an error.
- drone pose: shown by the share above 70 km/h against the drone's turn rate and speed during
  the track, and for tracks that are parked by image motion alone (D-510), whose true speed is
  zero whatever the pose says.
- measurement noise alone: each track is replayed as if the object stood still at its median
  ground point, with noise drawn from version 1's measured noise model for each frame
  (fixed seed). If that still twin also reads above 70 km/h, noise alone explains the track.
Each removal is also applied on top of the timing correction, since timing affects every track.
Output: results/auair/speed_causes_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerialrange.evaluate.auair_world import ground_offset_oblique  # noqa: E402
from aerialrange.geometry.kalman import GroundKalman  # noqa: E402
from aerialrange.utils.config import REPO_ROOT, load_config  # noqa: E402
from auair_moving import EDGES, MIN_MEAS, jitter_sigma  # noqa: E402

FAST_KMH = 70.0  # version 1's implausibility line (D-518)
SETTLE = 3  # filtered rows skipped at the start of a track, as version 1
DEP_BANDS = ((0, 20), (20, 40), (40, 60), (60, 91))
SEED = 20261108  # noise-only twins, fixed so the result repeats
LINK_IOU = 0.5  # detector box to annotation box, the evaluation threshold (D-302)


def iou_one_to_many(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """IoU of one ltwh box with many."""
    x1 = np.maximum(b[0], bs[:, 0])
    y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[0] + b[2], bs[:, 0] + bs[:, 2])
    y2 = np.minimum(b[1] + b[3], bs[:, 1] + bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = b[2] * b[3] + bs[:, 2] * bs[:, 3] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def truth_ids(tr: np.ndarray, by_f: dict[int, np.ndarray]) -> np.ndarray:
    """Annotation track id matched to each detector row (frame, id, l, t, w, h, ...): best IoU
    >= LINK_IOU among the annotation boxes of that frame (`by_f`: frame -> rows), else -1."""
    out = np.full(len(tr), -1)
    for i, r in enumerate(tr):
        cand = by_f.get(int(r[0]))
        if cand is None:
            continue
        iou = iou_one_to_many(r[2:6], cand[:, 2:6])
        j = int(np.argmax(iou))
        if iou[j] >= LINK_IOU:
            out[i] = int(cand[j, 1])
    return out


def cut_at_switches(ids: np.ndarray) -> list[np.ndarray]:
    """Row index pieces of a track, cut where the matched annotation id changes (unmatched
    rows stay with the piece before them)."""
    pieces, start, cur = [], 0, -1
    for i, t in enumerate(ids):
        if t < 0:
            continue
        if cur >= 0 and t != cur:
            pieces.append(np.arange(start, i))
            start = i
        cur = t
    pieces.append(np.arange(start, len(ids)))
    return pieces


def track_speed(t_s: np.ndarray, pts: np.ndarray, sig: np.ndarray, q: float) -> float | None:
    """Median filtered speed (m/s) of a track, as version 1 (auair_moving.filter_track), with
    the time of each measurement given in seconds."""
    if len(t_s) < SETTLE + 3:
        return None
    R = [np.eye(2) * s**2 for s in sig]
    kf = GroundKalman.from_two_points(pts[0], R[0], pts[1], R[1], t_s[1] - t_s[0], q)
    sp, last = [], t_s[1]
    for j in range(2, len(t_s)):
        kf.predict(t_s[j] - last)
        last = t_s[j]
        kf.update(pts[j], R[j])
        sp.append(kf.speed_heading()[0])
    sp = sp[SETTLE:]
    return float(np.median(sp)) if sp else None


class Clip:
    """Per-clip camera data: timestamps, tilt, heading, drift-corrected height, position."""

    def __init__(self, c: int, fr: pd.DataFrame, drift: dict) -> None:
        g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
        tp = REPO_ROOT / "cache" / "auair" / "tilt" / f"clip_{c:02d}_frames.csv"
        self.ok = tp.exists()
        if not self.ok:
            return
        self.tilt = pd.read_csv(tp)["tilt_deg"].to_numpy()
        d = drift.get(g["recording"][0])
        t = g["t_ms"].to_numpy()
        h0 = (d["a_m"] + d["b_m_per_min"] / 60 * t / 1000) if d else np.zeros(len(g))
        self.t_s = t / 1000.0
        self.h = g["alt_m"].to_numpy() + h0
        self.yaw = g["angle_psi"].to_numpy()
        self.n, self.e = g["n_m"].to_numpy(), g["e_m"].to_numpy()

    def ground(self, rows: np.ndarray, f: float, tilt: np.ndarray) -> tuple:
        """Ground points, slant range and line-of-sight depression of box bottom-centres."""
        k = rows[:, 0].astype(int) - 1
        u, v = rows[:, 2] + rows[:, 4] / 2, rows[:, 3] + rows[:, 5]
        dn, de = ground_offset_oblique(u, v, self.h[k], f, self.yaw[k], tilt)
        pts = np.column_stack([self.n[k] + dn, self.e[k] + de])
        dep = tilt + np.degrees(np.arctan((v - 540.0) / f))
        return pts, np.hypot(np.hypot(dn, de), self.h[k]), dep


def analyse_track(
    rows: np.ndarray,
    clip: Clip,
    f: float,
    rel: np.ndarray,
    q: float,
    tids: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    """Speeds of one track under each variant, plus its attributes. A variant that cannot be
    computed (for example the fixed tilt puts points above the horizon) is None; the track is
    dropped only if version 1's own variant cannot be computed."""
    k = rows[:, 0].astype(int) - 1
    ok = np.isfinite(clip.tilt[k])
    if ok.sum() < MIN_MEAS:
        return None
    rows, tids, k = rows[ok], tids[ok], k[ok]
    tilt = clip.tilt[k]
    out: dict = {}
    for tilt_name, tl in (("tilt", tilt), ("tilt_fixed", np.full(len(k), np.median(tilt)))):
        pts, slant, dep = clip.ground(rows, f, tl)
        band = np.clip(np.searchsorted(EDGES[:, 1], dep, side="right"), 0, len(rel) - 1)
        sig = rel[band] * slant
        m = np.isfinite(pts).all(axis=1) & np.isfinite(sig)
        if m.sum() < MIN_MEAS:
            if tilt_name == "tilt":
                return None
            out["frames+tilt_fixed"] = out["stamps+tilt_fixed"] = None
            continue
        fidx = rows[m, 0]
        times = (("frames", (fidx - fidx[0]) * 0.2), ("stamps", clip.t_s[k[m]]))
        for time_name, ts in times:
            out[f"{time_name}+{tilt_name}"] = track_speed(ts, pts[m], sig[m], q)
        if tilt_name != "tilt":
            continue
        out["depression_median"] = float(np.median(dep[m]))
        out["slant_median_m"] = float(np.median(slant[m]))
        out["measurements"] = int(m.sum())
        # the same track if the object stood still: its median point plus measurement noise
        # drawn from version 1's own noise model (D-518)
        still = np.median(pts[m], axis=0) + rng.normal(size=(int(m.sum()), 2)) * sig[m, None]
        for time_name, ts in times:
            out[f"{time_name}+noise_only"] = track_speed(ts, still, sig[m], q)
            pieces = cut_at_switches(tids[m])  # identity switches removed
            out[f"{time_name}+cut"] = [
                sp
                for p in pieces
                if len(p) >= MIN_MEAS
                and (sp := track_speed(ts[p], pts[m][p], sig[m][p], q)) is not None
            ]
        # frames missing from the recording within the track (not missed detections)
        dt = np.diff(clip.t_s[k[m]])
        out["dropped_frame_steps"] = int((np.round((dt - np.diff(fidx) * 0.2) * 1000) > 0).sum())
    t_all = clip.t_s[k]
    yaw = np.unwrap(clip.yaw[k])
    dts = np.diff(t_all)
    good = dts > 0
    if good.any():
        out["yaw_rate_p90_deg_s"] = float(
            np.percentile(np.degrees(np.abs(np.diff(yaw)[good] / dts[good])), 90)
        )
        step = np.hypot(np.diff(clip.n[k]), np.diff(clip.e[k]))
        out["drone_speed_median_ms"] = float(np.median(step[good] / dts[good]))
    else:
        out["yaw_rate_p90_deg_s"] = out["drone_speed_median_ms"] = 0.0
    out["max_tilt_step_deg"] = float(np.max(np.abs(np.diff(tilt)))) if len(tilt) > 1 else 0.0
    m_ids = tids[tids >= 0]
    out["switches"] = int((np.diff(m_ids) != 0).sum()) if len(m_ids) > 1 else 0
    out["matched_share"] = float((tids >= 0).mean())
    out["frames"] = rows[:, 0].astype(int)  # for the parked-in-the-image share, then dropped
    return out


def parked_share(frames: np.ndarray, segs: pd.DataFrame | None, tid: int) -> float:
    """Share of a track's frames inside segments found parked from image motion alone
    (scripts/auair_stationary.py, D-510; no pose, height or GPS is used there)."""
    if segs is None or not len(segs):
        return 0.0
    inside = np.zeros(len(frames), bool)
    for _, sg in segs[segs["track"] == tid].iterrows():
        inside |= (frames >= sg["first_frame"]) & (frames <= sg["last_frame"])
    return float(inside.mean())


def share_fast(speeds: list[float | None]) -> dict:
    s = np.array([x for x in speeds if x is not None], float) * 3.6
    return {
        "tracks": int(len(s)),
        "share_above_70_kmh": float((s > FAST_KMH).mean()) if len(s) else None,
    }


def share_fast_pieces(pieces: list[list[float]]) -> dict:
    """Pieces after cutting at switches, each counted as a track."""
    return share_fast([s for p in pieces for s in p])


def run(run: str, truth_run: str | None) -> dict:
    au_cfg = load_config("auair")
    f = float(au_cfg.require("focal_px"))
    calib = set(au_cfg.require("calib_recordings"))
    q = float(load_config("geometry").require("kalman_q"))
    drift = json.loads((REPO_ROOT / "results/auair/camera_scale.json").read_text())[
        "altimeter_drift"
    ]
    fr = pd.read_csv(REPO_ROOT / "cache/auair/frames.csv", dtype={"recording": str})
    rec_of = fr.groupby("clip")["recording"].first()
    cal_clips = [int(c) for c in rec_of.index if rec_of[c] in calib]
    rel = jitter_sigma("B2_yolo26n_960", {"focal_px": f}, drift, fr, cal_clips)  # D-518 noise
    tracks = []
    for c in sorted(int(x) for x in rec_of.index):
        clip = Clip(c, fr, drift)
        if not clip.ok:
            continue
        p = REPO_ROOT / "cache" / "auair" / "runs" / run / f"clip_{c:02d}_vehicle.txt"
        if not p.exists() or p.stat().st_size == 0:
            continue
        tr = np.loadtxt(p, delimiter=",", ndmin=2)
        sp_ = REPO_ROOT / "cache" / "auair" / "stationary" / run / f"clip_{c:02d}.csv"
        segs = pd.read_csv(sp_) if sp_.exists() and sp_.stat().st_size > 5 else None
        by_f: dict[int, np.ndarray] = {}
        if truth_run:
            tp = REPO_ROOT / "cache" / "auair" / "runs" / truth_run / f"clip_{c:02d}_vehicle.txt"
            if tp.exists() and tp.stat().st_size:
                truth = np.loadtxt(tp, delimiter=",", ndmin=2)
                by_f = {int(f): truth[truth[:, 0] == f] for f in np.unique(truth[:, 0])}
        for tid in np.unique(tr[:, 1]):
            r = tr[tr[:, 1] == tid]
            r = r[np.argsort(r[:, 0])]
            rng = np.random.default_rng([SEED, c, int(tid)])
            a = analyse_track(r, clip, f, rel, q, truth_ids(r, by_f), rng)
            if a is not None:
                a["parked_in_image_share"] = parked_share(a.pop("frames"), segs, int(tid))
                tracks.append({"clip": c, "track": int(tid), **a})
    return {"tracks": tracks}


def summarise(tracks: list[dict]) -> dict:
    out: dict = {}
    variants = ("tilt", "tilt_fixed", "noise_only")
    for v in (f"{t}+{x}" for x in variants for t in ("frames", "stamps")):
        out[v] = share_fast([t[v] for t in tracks])
    for v in ("frames+cut", "stamps+cut"):
        out[v] = share_fast_pieces([t[v] for t in tracks])
    fast = np.array([(t["frames+tilt"] or 0) * 3.6 > FAST_KMH for t in tracks])
    fast_ts = np.array([(t["stamps+tilt"] or 0) * 3.6 > FAST_KMH for t in tracks])
    noise = np.array([(t["frames+noise_only"] or 0) * 3.6 > FAST_KMH for t in tracks])
    out["fast_v1_whose_still_twin_is_fast"] = float(noise[fast].mean()) if fast.any() else None

    def split(name: str, mask: np.ndarray) -> dict:
        return {
            "tracks": int(mask.sum()),
            "share_fast_v1": float(fast[mask].mean()) if mask.any() else None,
            "share_fast_timestamps": float(fast_ts[mask].mean()) if mask.any() else None,
            "share_fast_if_still": float(noise[mask].mean()) if mask.any() else None,
            "median_slant_m": float(np.median(slant[mask])) if mask.any() else None,
        }

    dep = np.array([t["depression_median"] for t in tracks])
    slant = np.array([t["slant_median_m"] for t in tracks])
    out["by_depression"] = {f"{a}-{b}": split("dep", (dep >= a) & (dep < b)) for a, b in DEP_BANDS}
    drop = np.array([t["dropped_frame_steps"] for t in tracks])
    out["by_dropped_frames"] = {"none": split("d", drop == 0), "some": split("d", drop > 0)}
    sw = np.array([t["switches"] for t in tracks])
    out["by_switches"] = {"none": split("s", sw == 0), "some": split("s", sw > 0)}
    pk = np.array([t["parked_in_image_share"] for t in tracks])
    out["by_parked_in_image"] = {
        "mostly_parked": split("p", pk >= 0.5),
        "moving_or_unknown": split("p", pk < 0.5),
    }
    for key in ("yaw_rate_p90_deg_s", "drone_speed_median_ms"):
        x = np.array([t[key] for t in tracks])
        qs = np.percentile(x, [25, 50, 75])
        edges = [-np.inf, *qs, np.inf]
        out[f"by_{key}"] = {
            "quartile_edges": qs.tolist(),
            **{f"q{i + 1}": split(key, (x > edges[i]) & (x <= edges[i + 1])) for i in range(4)},
        }
    ts = np.array([t["max_tilt_step_deg"] for t in tracks])
    med = float(np.median(ts))
    out["by_max_tilt_step"] = {
        "median_deg": med,
        "below_median": split("t", ts <= med),
        "above_median": split("t", ts > med),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--truth-run", default="truth")
    args = ap.parse_args(argv)
    det = run(args.run, args.truth_run)
    ann = run(args.truth_run, None)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "truth_run": args.truth_run,
        "fast_kmh": FAST_KMH,
        "detector_tracks": summarise(det["tracks"]),
        "annotation_tracks": summarise(ann["tracks"]),
        "note": "No ground-truth speeds: shares of implausible speeds only.",
    }
    for name, res in (("detector", det), ("annotation", ann)):
        pd.DataFrame(
            [{k: v for k, v in t.items() if "+cut" not in k} for t in res["tracks"]]
        ).to_csv(REPO_ROOT / "cache" / "auair" / f"speed_causes_{args.run}_{name}.csv", index=False)
    dest = REPO_ROOT / "results" / "auair" / f"speed_causes_{args.run}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("detector_tracks", "annotation_tracks")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
