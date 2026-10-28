"""Export one AU-AIR clip as pipeline inputs: a frames folder and a flight-data CSV.

Usage: python scripts/auair_export_flight.py --clip 46 [--out cache/auair/flights]

Frames are copied in time order to <out>/clip_NN/frames/ (AU-AIR images are not redistributed;
the folder is in cache/, D-323). metadata.csv has one row per frame: frame (1-based), t_s,
n_m / e_m (local north / east from GPS, clip origin), h_m (altitude corrected for the altimeter
offset where one was measured, D-509), yaw_rad (`angle_psi`), tilt_deg (tilt from motion,
D-515; NaN where none). Then: aerialrange run --source <out>/clip_NN/frames
--metadata <out>/clip_NN/metadata.csv --out <run folder>.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

from aerialrange.utils.config import REPO_ROOT, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--out", default="cache/auair/flights")
    args = ap.parse_args(argv)
    data = load_config("data")
    root = REPO_ROOT / data.require("data_root") / data.require("auair_subdir")
    fr = pd.read_csv(REPO_ROOT / "cache/auair/frames.csv", dtype={"recording": str})
    g = fr[fr["clip"] == args.clip].sort_values("t_ms").reset_index(drop=True)
    drift = json.loads((REPO_ROOT / "results/auair/camera_scale.json").read_text())[
        "altimeter_drift"
    ]
    d = drift.get(g["recording"][0])
    h0 = (d["a_m"] + d["b_m_per_min"] / 60 * g["t_ms"] / 1000).to_numpy() if d else np.zeros(len(g))
    tp = REPO_ROOT / "cache/auair/tilt" / f"clip_{args.clip:02d}_frames.csv"
    tilt = pd.read_csv(tp)["tilt_deg"].to_numpy() if tp.exists() else np.full(len(g), np.nan)
    out = REPO_ROOT / args.out / f"clip_{args.clip:02d}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(g["image_path"], start=1):
        shutil.copy2(root / p, out / "frames" / f"{k:06d}.jpg")
    pd.DataFrame(
        {
            "frame": np.arange(1, len(g) + 1),
            "t_s": (g["t_ms"] - g["t_ms"][0]).to_numpy() / 1000,
            "n_m": g["n_m"],
            "e_m": g["e_m"],
            "h_m": g["alt_m"].to_numpy() + h0,
            "yaw_rad": g["angle_psi"],
            "tilt_deg": tilt,
        }
    ).to_csv(out / "metadata.csv", index=False)
    n_tilt = int(np.isfinite(tilt).sum())
    print(f"clip {args.clip}: {len(g)} frames, tilt known for {n_tilt}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
