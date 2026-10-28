"""Repeatability of ground positions of stationary vehicles on straight-down AU-AIR clips
(D-511).

Usage: python scripts/auair_repeatability.py --run B2_yolo26n_960 --split calibration
       python scripts/auair_repeatability.py --run B2_yolo26n_960 --split held_out --final

Per stationary segment (scripts/auair_stationary.py), each frame's box centre is turned into a
ground position: drone position (GPS, local NED) + the pixel's ground offset
(`evaluate/auair_world.py`) at the drift-corrected height (results/auair/camera_scale.json) and
the fitted focal length (configs/auair.yaml). Spread = RMS distance about the segment mean.
Hypotheses (calibration only): 8 mounts (image rotation and mirror) x camera model:
H-stabilised (body tilt does not move the image) or H-rigid (the view tilts with the body;
4 sign choices x which of angle_phi / angle_theta is pitch). The calibration split selects the
hypothesis with the smallest median hover spread; the held-out split applies it unchanged,
once, with --final. Hover frames: horizontal speed below `HOVER_SPEED_MS` for at least
`HOVER_MIN_S` (D-511); the all-flight spread also includes GPS drift while moving.
Output: results/auair/repeatability_<split>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aerialrange.evaluate.auair_world import HYPOTHESES, ground_offset_nadir, spread_rms
from aerialrange.utils.config import REPO_ROOT, load_config

# D-511: velocity noise in the hover mode is about 0.04 m/s per axis (robust sigma, calibration
# recordings); 99 % of the speeds of 2-D Gaussian noise of that size lie below 3.03 sigma.
HOVER_SPEED_MS = 0.12
HOVER_MIN_S = 4.0
MIN_FRAMES = 20
N_BOOT = 2000
RIGID = [("stabilised", None)] + [
    ("rigid", (sp, sr, swap)) for sp, sr, swap in itertools.product((1, -1), (1, -1), (False, True))
]


def hover_mask(speed: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Frames inside runs of speed below HOVER_SPEED_MS lasting at least HOVER_MIN_S."""
    ok = speed < HOVER_SPEED_MS
    out = np.zeros(len(ok), bool)
    i = 0
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


def positions(seg: pd.DataFrame, f: float, rot: int, mirror: bool, model: tuple) -> np.ndarray:
    dn, de = ground_offset_nadir(
        seg["u"].to_numpy(),
        seg["v"].to_numpy(),
        seg["h"].to_numpy(),
        f,
        seg["yaw"].to_numpy(),
        rot,
        mirror,
    )
    if model[0] == "rigid":
        sp, sr, swap = model[1]
        pitch, roll = (seg["phi"], seg["theta"]) if swap else (seg["theta"], seg["phi"])
        fwd = sp * seg["h"].to_numpy() * np.tan(pitch.to_numpy())
        right = sr * seg["h"].to_numpy() * np.tan(roll.to_numpy())
        y = seg["yaw"].to_numpy()
        dn = dn + np.cos(y) * fwd - np.sin(y) * right
        de = de + np.sin(y) * fwd + np.cos(y) * right
    return np.column_stack([seg["n"].to_numpy() + dn, seg["e"].to_numpy() + de])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["calibration", "held_out"])
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "held_out" and not args.final:
        print("Held-out clips run once, after the hypothesis is frozen. Use --final.")
        return 1
    au = load_config("auair")
    f = float(au.require("focal_px"))
    calib = set(au.require("calib_recordings"))
    scale = json.loads((REPO_ROOT / "results" / "auair" / "camera_scale.json").read_text())
    drift = scale["altimeter_drift"]
    fr = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv", dtype={"recording": str})
    lab = pd.read_csv(REPO_ROOT / "results" / "auair" / "clip_labels.csv")
    down = set(lab.loc[lab["view"] == "down", "clip"])
    rec_of = fr.groupby("clip")["recording"].first()
    clips = sorted(c for c in down if (rec_of[c] in calib) == (args.split == "calibration"))
    tracks_dir = REPO_ROOT / "cache" / "auair" / "runs" / args.run
    seg_dir = REPO_ROOT / "cache" / "auair" / "stationary" / args.run
    segs = []
    for c in clips:
        g = fr[fr["clip"] == c].sort_values("t_ms").reset_index(drop=True)
        rec = g["recording"][0]
        if rec in drift:
            d = drift[rec]
            h0 = d["a_m"] + d["b_m_per_min"] / 60 * g["t_ms"] / 1000
            h0_note = d["model"]
        else:
            h0, h0_note = 0.0, "no climb in this recording: altimeter used as recorded"
        g = g.assign(
            h=g["alt_m"] + h0,
            speed=np.hypot(g["linear_x"], g["linear_y"]),
            hover=hover_mask(
                np.hypot(g["linear_x"], g["linear_y"]).to_numpy(), g["t_ms"].to_numpy() / 1000
            ),
        )
        p = seg_dir / f"clip_{c:02d}.csv"
        if not p.exists() or p.stat().st_size < 5:
            continue
        st = pd.read_csv(p)
        tr = np.loadtxt(tracks_dir / f"clip_{c:02d}_vehicle.txt", delimiter=",", ndmin=2)
        for _, s in st.iterrows():
            rows = tr[
                (tr[:, 1] == s["track"])
                & (tr[:, 0] >= s["first_frame"])
                & (tr[:, 0] <= s["last_frame"])
            ]
            k = rows[:, 0].astype(int) - 1
            seg = pd.DataFrame(
                {
                    "u": rows[:, 2] + rows[:, 4] / 2,
                    "v": rows[:, 3] + rows[:, 5] / 2,
                    "h": g["h"].to_numpy()[k],
                    "yaw": g["angle_psi"].to_numpy()[k],
                    "phi": g["angle_phi"].to_numpy()[k],
                    "theta": g["angle_theta"].to_numpy()[k],
                    "n": g["n_m"].to_numpy()[k],
                    "e": g["e_m"].to_numpy()[k],
                    "hover": g["hover"].to_numpy()[k],
                }
            )
            segs.append(
                {
                    "clip": int(c),
                    "track": int(s["track"]),
                    "seg": seg,
                    "h0_model": h0_note,
                    "alt_m": float(g["alt_m"].to_numpy()[k].mean()),
                }
            )
    hyps = [(r, m, model) for (r, m) in HYPOTHESES for model in RIGID]
    if args.split == "held_out":
        frozen = json.loads(
            (REPO_ROOT / "results" / "auair" / "repeatability_calibration.json").read_text()
        )["selected"]
        hyps = [
            (
                frozen["rot"],
                frozen["mirror"],
                (frozen["model"], tuple(frozen["rigid"]) if frozen["rigid"] else None),
            )
        ]

    def evaluate(rot: int, mirror: bool, model: tuple) -> dict:
        allf, hov = [], []
        for s in segs:
            pts = positions(s["seg"], f, rot, mirror, model)
            allf.append(spread_rms(pts))
            hm = s["seg"]["hover"].to_numpy()
            hov.append(spread_rms(pts[hm]) if hm.sum() >= MIN_FRAMES else np.nan)
        return {"all": np.array(allf), "hover": np.array(hov)}

    table = []
    for rot, mirror, model in hyps:
        r = evaluate(rot, mirror, model)
        table.append(
            {
                "rot": rot,
                "mirror": mirror,
                "model": model[0],
                "rigid": list(model[1]) if model[1] else None,
                "median_spread_hover_m": float(np.nanmedian(r["hover"]))
                if np.isfinite(r["hover"]).any()
                else None,
                "median_spread_all_m": float(np.median(r["all"])) if len(r["all"]) else None,
                "_r": r,
            }
        )

    def key(t: dict) -> float:
        return t["median_spread_hover_m"] if t["median_spread_hover_m"] is not None else np.inf

    best = min(table, key=key)
    rng = np.random.default_rng(load_config("project").require("seed"))

    def boot(x: np.ndarray) -> list[float] | None:
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return None
        b = [np.median(rng.choice(x, len(x))) for _ in range(N_BOOT)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    r = best.pop("_r")
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-511",
        "run": args.run,
        "split": args.split,
        "clips": [int(c) for c in clips],
        "focal_px": f,
        "hover": {"speed_ms": HOVER_SPEED_MS, "min_s": HOVER_MIN_S, "min_frames": MIN_FRAMES},
        "n_segments": len(segs),
        "n_segments_with_hover": int(np.isfinite(r["hover"]).sum()),
        "selected": {k: best[k] for k in ("rot", "mirror", "model", "rigid")},
        "spread_hover_m": {
            "median": best["median_spread_hover_m"],
            "ci95_segment_bootstrap": boot(r["hover"]),
            "p90": float(np.nanpercentile(r["hover"], 90))
            if np.isfinite(r["hover"]).any()
            else None,
        },
        "spread_all_flight_m": {
            "median": best["median_spread_all_m"],
            "ci95_segment_bootstrap": boot(r["all"]),
            "note": "includes GPS drift while the drone moves",
        },
        "by_altitude": {
            band: float(np.nanmedian(r["hover"][m])) if np.isfinite(r["hover"][m]).any() else None
            for band, m in (
                ("<15 m", np.array([s["alt_m"] < 15 for s in segs])),
                ("15-25 m", np.array([15 <= s["alt_m"] < 25 for s in segs])),
                (">=25 m", np.array([s["alt_m"] >= 25 for s in segs])),
            )
        }
        if segs
        else {},
        "hypotheses": [{k: v for k, v in t.items() if k != "_r"} for t in table],
        "caveats": "Principal point assumed at the image centre; yaw assumed clockwise from north; "
        "segment spreads within one clip are correlated; GPS horizontal error enters the "
        "all-flight spread.",
    }
    dest = REPO_ROOT / "results" / "auair" / f"repeatability_{args.split}.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "n_segments",
                    "n_segments_with_hover",
                    "selected",
                    "spread_hover_m",
                    "spread_all_flight_m",
                    "by_altitude",
                )
            },
            indent=1,
        )
    )
    print("top hypotheses:")
    for t in sorted(out["hypotheses"], key=lambda t: t["median_spread_hover_m"] or np.inf)[:6]:
        print(" ", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
