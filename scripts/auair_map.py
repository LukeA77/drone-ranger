"""Map layer for an AU-AIR clip run (Phase 12): replay its tracks with the current stage, then
write map.geojson and map.html.

Usage: python scripts/auair_map.py --clip 35 --source cache/runs/v2_clip35 [--config run]
       [--name v2_clip35_map]

The source run's tracks.csv (boxes and track numbers) is replayed through the range and speed
stage (`pipeline.range_speed.replay`) with the config given, so its rows carry the ground point
and covariance columns; the clip's flight data is cache/auair/flights/clip_<nn>/metadata.csv.
The local frame's origin is the clip's first GPS fix (cache/auair/frames.csv, as
`data/auair.py` builds the local positions). AU-AIR's altitude is not an ellipsoidal height;
an error of tens of metres there moves map positions by under 1 cm per kilometre.
Writes cache/runs/<name>/{tracks.csv, map.geojson, map.html}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aerialrange.pipeline.geomap import write_map_from_frames
from aerialrange.pipeline.range_speed import RangeSpeedStage, ranging_model, replay
from aerialrange.utils.config import REPO_ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--config", default="run")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = pd.read_csv(
        REPO_ROOT / "cache" / "auair" / "flights" / f"clip_{args.clip:02d}" / "metadata.csv"
    )
    frames = pd.read_csv(REPO_ROOT / "cache" / "auair" / "frames.csv")
    first = frames[frames["clip"] == args.clip].sort_values("t_ms").iloc[0]
    if abs(first["n_m"]) > 1e-9 or abs(first["e_m"]) > 1e-9:
        raise SystemExit("the clip's first fix is not its local origin")
    origin = (float(first["lat_deg"]), float(first["lon_deg"]), float(first["alt_m"]))
    src = pd.read_csv(REPO_ROOT / args.source / "tracks.csv")
    stage = RangeSpeedStage.from_config(cfg.require)
    tracks = replay(src, meta.set_index("frame"), stage)
    name = args.name or f"{Path(args.source).name}_map"
    out = REPO_ROOT / "cache" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(out / "tracks.csv", index=False)
    gj = write_map_from_frames(tracks, meta, origin, ranging_model(cfg.require), out, name)
    kinds = pd.Series([f["properties"]["kind"] for f in gj["features"]]).value_counts()
    print(kinds.to_dict())
    print("origin", origin, "wrote", out / "map.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
