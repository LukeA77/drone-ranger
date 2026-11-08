"""AU-AIR vehicle speeds through the pipeline's ranging and speed stage, settings compared on
identical tracks (D-735).

Usage: python scripts/auair_speed_replay.py --run B2_yolo26n_960 --variant <name>
       --config run_accurate [--set kalman.meas_model=propagated ...]

Each clip's tracks (cache/auair/runs/<run>/clip_NN_vehicle.txt) are replayed through
`pipeline/range_speed.py` with per-frame flight data built as in scripts/auair_moving.py:
GPS position, heading (`angle_psi`), altimeter height with the drift correction of the
calibration recordings (D-508), the tilt estimated from ground motion (scripts/auair_tilt.py)
and timestamps. The tilt's uncertainty per frame (`tilt_sigma_deg`) is the standard error of
the median of the estimates it was taken from, 1.2533 x spread / sqrt(n); neighbouring
frames share estimates, so it is a lower bound on the error and its errors are correlated in
time. Per track: the median of the filtered speed after the first 3 updates (as version 1),
and whether any speed passes the display rule. Reports the share of vehicle tracks above
70 km/h, overall and by line-of-sight depression, against version 1 (D-518, D-734).
AU-AIR has no true speeds: plausibility only.
Output: results/auair/speed_replay_<run>__<variant>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerialrange.pipeline.range_speed import RangeSpeedStage, replay  # noqa: E402
from aerialrange.utils.config import REPO_ROOT, load_config  # noqa: E402
from syndrone_refilter import parse_sets  # noqa: E402

FAST_KMH = 70.0
SETTLE = 3
MIN_ROWS = 10  # tracks with at least this many filtered speeds, as version 1's MIN_MEAS
DEP_BANDS = ((0, 10), (10, 20), (20, 40), (40, 91))
MEDIAN_SE = 1.2533  # standard error of a median = 1.2533 sigma / sqrt(n) (normal errors)


def clip_meta(c: int, fr: pd.DataFrame, drift: dict) -> pd.DataFrame | None:
    g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
    tp = REPO_ROOT / "cache" / "auair" / "tilt" / f"clip_{c:02d}_frames.csv"
    if not tp.exists():
        return None
    tl = pd.read_csv(tp)
    d = drift.get(g["recording"][0])
    t = g["t_ms"].to_numpy()
    h0 = (d["a_m"] + d["b_m_per_min"] / 60 * t / 1000) if d else np.zeros(len(g))
    with np.errstate(divide="ignore", invalid="ignore"):
        tsig = MEDIAN_SE * tl["spread_deg"].to_numpy() / np.sqrt(tl["n_est"].to_numpy())
    return pd.DataFrame(
        {
            "frame": np.arange(1, len(g) + 1),
            "t_s": t / 1000.0,
            "n_m": g["n_m"].to_numpy(),
            "e_m": g["e_m"].to_numpy(),
            "h_m": g["alt_m"].to_numpy() + h0,
            "yaw_rad": g["angle_psi"].to_numpy(),
            "tilt_deg": tl["tilt_deg"].to_numpy(),
            "tilt_sigma_deg": tsig,
        }
    ).set_index("frame")


def track_table(c: int, run: str) -> pd.DataFrame | None:
    p = REPO_ROOT / "cache" / "auair" / "runs" / run / f"clip_{c:02d}_vehicle.txt"
    if not p.exists() or p.stat().st_size == 0:
        return None
    a = np.loadtxt(p, delimiter=",", ndmin=2)
    df = pd.DataFrame(a[:, :6], columns=["frame", "track_id", "left", "top", "width", "height"])
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["class"] = "vehicle"
    return df


def per_track(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for tid, g in rows.groupby("track_id"):
        g = g[g["valid"] & np.isfinite(g["speed_ms"])].sort_values("frame")
        sp = g["speed_ms"].to_numpy()[SETTLE:]
        if len(sp) < MIN_ROWS - SETTLE:
            continue
        out.append(
            {
                "track_id": int(tid),
                "speed_median_kmh": float(np.median(sp) * 3.6),
                "depression_median": float(np.median(g["depression_deg"])),
                "ever_shown": bool(g["speed_shown"].any()),
                "shown_median_kmh": float(np.median(g.loc[g["speed_shown"], "speed_ms"]) * 3.6)
                if g["speed_shown"].any()
                else np.nan,
            }
        )
    return pd.DataFrame(out)


def summary(pt: pd.DataFrame) -> dict[str, Any]:
    fast = pt["speed_median_kmh"] > FAST_KMH
    shown = pt["ever_shown"]
    res: dict[str, Any] = {
        "tracks": int(len(pt)),
        "share_above_70_kmh": float(fast.mean()) if len(pt) else None,
        "tracks_with_a_shown_speed": int(shown.sum()),
        "share_of_shown_above_70_kmh": float((pt.loc[shown, "shown_median_kmh"] > FAST_KMH).mean())
        if shown.any()
        else None,
        "by_depression": {},
    }
    for a, b in DEP_BANDS:
        m = (pt["depression_median"] >= a) & (pt["depression_median"] < b)
        res["by_depression"][f"{a}-{b}"] = {
            "tracks": int(m.sum()),
            "share_above_70_kmh": float(fast[m].mean()) if m.any() else None,
        }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default="run_accurate")
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ov = parse_sets(args.sets)

    def get(k: str) -> Any:
        base = cfg.require(k)
        return {**base, **ov.get(k, {})} if isinstance(base, dict) else ov.get(k, base)

    au = load_config("auair")
    calib = set(au.require("calib_recordings"))
    drift = json.loads((REPO_ROOT / "results/auair/camera_scale.json").read_text())[
        "altimeter_drift"
    ]
    fr = pd.read_csv(REPO_ROOT / "cache/auair/frames.csv", dtype={"recording": str})
    rec_of = fr.groupby("clip")["recording"].first()
    tables = []
    for c in sorted(int(x) for x in rec_of.index):
        meta, tracks = clip_meta(c, fr, drift), track_table(c, args.run)
        if meta is None or tracks is None:
            continue
        rows = replay(tracks, meta, RangeSpeedStage.from_config(get))
        pt = per_track(rows)
        if len(pt):
            pt["clip"] = c
            pt["split"] = "calibration" if rec_of[c] in calib else "held_out"
            tables.append(pt)
    pt = pd.concat(tables, ignore_index=True)
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-735",
        "run": args.run,
        "variant": args.variant,
        "config": args.config,
        "overrides": ov,
        "all": summary(pt),
        "calibration": summary(pt[pt["split"] == "calibration"]),
        "held_out": summary(pt[pt["split"] == "held_out"]),
        "note": "No ground-truth speeds: plausibility only.",
    }
    dest = REPO_ROOT / "results" / "auair" / f"speed_replay_{args.run}__{args.variant}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    pt.to_csv(
        REPO_ROOT / "cache" / "auair" / f"speed_replay_{args.run}__{args.variant}.csv", index=False
    )
    print(json.dumps(out["all"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
