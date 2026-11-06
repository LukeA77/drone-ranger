"""Speed and heading truth for every SynDrone actor and frame of one town.

Reads `cache/syndrone/actors/<town>.npz` (scripts/syndrone_actor_tracks.py) and writes
`cache/syndrone/speed/<town>.csv.gz` (one row per actor and frame) plus a summary of the
truth's own uncertainty to `results/syndrone/development/speed_truth_<town>.json`
(development towns only; other towns' summaries go to `cache/`).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from aerialrange.data.syndrone import FPS, PERSON
from aerialrange.evaluate.syndrone_speed import jitter_sigma, savgol_velocity, speed_truth
from aerialrange.utils.config import REPO_ROOT, load_config

ACT = REPO_ROOT / "cache" / "syndrone" / "actors"
OUT = REPO_ROOT / "cache" / "syndrone" / "speed"


def split_of(town: str) -> str:
    s = json.loads((REPO_ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    return next(k for k, v in s["syndrone"]["splits"].items() if town in v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    args = ap.parse_args(argv)
    cfg = load_config("syndrone")
    window = int(cfg.require("speed_window_frames"))
    vw, vv = float(cfg.require("vmax_walker_ms")), float(cfg.require("vmax_vehicle_ms"))
    hmin = float(cfg.require("heading_min_speed_ms"))
    dt = 1.0 / FPS
    z = np.load(ACT / f"{args.town}.npz")
    tables, jit, bias, n_cut, n_samples = [], [], [], 0, 0
    for i, tag in zip(z["ids"], z["tags"], strict=True):
        r = z[f"a{i}"]
        f, p = r[:, 0].astype(int), r[:, 1:4]
        st = speed_truth(f, p, dt, window, vw if tag == PERSON else vv, sigma_pos=0.0)
        n_cut += int(st["segment"].max())  # cuts inside this track
        n_samples += len(f)
        moving = st["speed"] > hmin
        if moving.sum() > 50:
            jit.append(jitter_sigma(p[moving, :2]))
            v5 = np.hypot(*savgol_velocity(p[:, :2], window + 2, dt).T)
            ok = moving & ~np.isnan(v5)
            bias.append(np.abs(st["speed"][ok] - v5[ok]))
        head = np.where(st["speed"] > hmin, st["heading_deg"], np.nan)
        tables.append(
            pd.DataFrame(
                {
                    "id": int(i),
                    "tag": int(tag),
                    "frame": f,
                    "x": p[:, 0],
                    "y": p[:, 1],
                    "z": p[:, 2],
                    "speed_ms": st["speed"],
                    "heading_deg": head,
                    "segment": st["segment"],
                }
            )
        )
    jit_p90 = float(np.nanpercentile(jit, 90)) if jit else float("nan")
    w = np.arange(-(window // 2), window // 2 + 1)
    sig_v = jit_p90 * np.sqrt(np.sum((w / np.sum(w * w)) ** 2)) / dt
    df = pd.concat(tables, ignore_index=True)
    df["speed_sigma_ms"] = np.where(df["speed_ms"].notna(), sig_v, np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{args.town}.csv.gz", index=False)
    b = np.concatenate(bias) if bias else np.zeros(0)
    summary = {
        "town": args.town,
        "split": split_of(args.town),
        "window_frames": window,
        "vmax_walker_ms": vw,
        "vmax_vehicle_ms": vv,
        "n_actors": int(len(z["ids"])),
        "n_samples": int(n_samples),
        "n_cuts": n_cut,
        "share_samples_with_speed": float(df["speed_ms"].notna().mean()),
        "position_jitter_m": {"median": float(np.nanmedian(jit)) if jit else None, "p90": jit_p90},
        "speed_sigma_from_jitter_ms": float(sig_v),
        "window_sensitivity_ms": {
            "note": f"|speed(window {window}) - speed(window {window + 2})| on moving samples; "
            "an upper bound on the bias from acceleration changes",
            "median": float(np.median(b)) if len(b) else None,
            "p99": float(np.percentile(b, 99)) if len(b) else None,
        },
    }
    dest = (
        (REPO_ROOT / "results" / "syndrone" / "development")
        if summary["split"] == "development"
        else (REPO_ROOT / "cache" / "syndrone" / "speed")
    )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"speed_truth_{args.town}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
