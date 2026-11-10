"""Search-and-rescue measures of a SynDrone evaluation (D-749).

Usage: python scripts/syndrone_search_metrics.py --name <run> [--split development]

For a search over open country, what matters is how many false alarms a crew is sent to per
area searched, and whether each person in view was found at all. From the per-sequence search
counts of scripts/syndrone_evaluate.py (results/syndrone/<split>/eval_<run>.json) and each
sequence's camera poses, this reports per group (person, vehicle):
- ground searched: the union over the run's frames of the ground the camera saw at a usable
  angle, the same outline the map draws as searched ground (pipeline/geomap.py: image rows whose
  line of sight is at least the lowest reliable depression below the horizon), measured on a
  0.5 m grid;
- false alarms per km2 searched: output tracks never matched to a valid truth box in any frame
  they were kept (after ignore regions are removed), divided by the ground searched;
- found share: valid truth objects matched by a track in at least one frame;
pooled over sequences, with 95 % bootstrap intervals that resample whole sequences, and per
height. Writes results/syndrone/<split>/search_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aerialrange.data.syndrone import syndrone_camera
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.evaluate.syndrone_eval import GROUPS
from aerialrange.pipeline.geomap import footprint_ring, min_reliable_depression_deg
from aerialrange.pipeline.range_speed import ranging_model
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syndrone_evaluate import N_BOOT, RUNS  # noqa: E402
from syndrone_refilter import flight_meta  # noqa: E402

GRID_M = 0.5  # raster cell for the union of footprints
IMAGE_W, IMAGE_H = 1920, 1080  # SynDrone frames


def searched_area_km2(meta: pd.DataFrame, model: object, min_dep: float, frames: set[int]) -> float:
    rings = []
    for f, pose in meta.iterrows():
        if int(f) not in frames:
            continue
        r = footprint_ring(pose, model, IMAGE_W, IMAGE_H, min_dep)
        if r is not None:
            rings.append(r)
    if not rings:
        return 0.0
    allp = np.vstack(rings)
    lo = allp.min(axis=0)
    size = np.ceil((allp.max(axis=0) - lo) / GRID_M).astype(int) + 2
    mask = np.zeros((size[0], size[1]), np.uint8)
    for r in rings:
        pts = np.round((r - lo) / GRID_M).astype(np.int32)[:, ::-1]  # (east, north) = (x, y)
        cv2.fillPoly(mask, [pts], 1)
    return float(mask.sum()) * GRID_M**2 / 1e6


def rates(counts: list[dict], area: list[float]) -> dict:
    fa = sum(c["n_false_alarm_tracks"] for c in counts)
    a = sum(area)
    n_obj = sum(c["n_truth_objects"] for c in counts)
    return {
        "ground_searched_km2": a,
        "n_tracks": sum(c["n_tracks"] for c in counts),
        "n_false_alarm_tracks": fa,
        "false_alarms_per_km2": fa / a if a > 0 else float("nan"),
        "n_truth_objects": n_obj,
        "n_truth_found": sum(c["n_truth_found"] for c in counts),
        "found_share": sum(c["n_truth_found"] for c in counts) / n_obj if n_obj else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="development")
    args = ap.parse_args(argv)
    ev_path = REPO_ROOT / "results" / "syndrone" / args.split / f"eval_{args.name}.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    cfg = load_config("run")
    cam = syndrone_camera()
    camera = {"focal_px": cam.fx, "cx": cam.cx, "cy": cam.cy}
    model = ranging_model(lambda k: camera if k == "camera" else cfg.require(k))
    min_dep = min_reliable_depression_deg(model)
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    area: dict[str, float] = {}
    for seq in ev["sequences"]:
        town, h = seq.rsplit("_h", 1)
        # the frames the run covered, as the evaluation counts them (1 to the last frame with a
        # track or a detection)
        run = RUNS / args.name / seq
        last = max(
            int(pd.read_csv(run / f, usecols=["frame"])["frame"].max())
            for f in ("tracks.csv", "detections.csv")
        )
        frames = set(range(1, last + 1))
        area[seq] = searched_area_km2(flight_meta(store, town, int(h)), model, min_dep, frames)
        print(f"{seq}: {area[seq]:.4f} km2 searched", flush=True)
    result: dict = {
        "name": args.name,
        "split": args.split,
        "min_depression_deg": min_dep,
        "grid_m": GRID_M,
        "n_boot": N_BOOT,
        "ground_searched_km2_per_sequence": area,
    }
    for g in GROUPS:
        per = ev[g]["search_counts_per_sequence"]
        seqs = list(per)
        res = {"pooled": rates([per[s] for s in seqs], [area[s] for s in seqs])}
        rng = np.random.default_rng(0)
        boot: dict[str, list[float]] = {"false_alarms_per_km2": [], "found_share": []}
        for _ in range(N_BOOT):
            pick = [seqs[i] for i in rng.integers(0, len(seqs), len(seqs))]
            r = rates([per[s] for s in pick], [area[s] for s in pick])
            for k in boot:
                if np.isfinite(r[k]):
                    boot[k].append(r[k])
        res["ci95_resample_sequences"] = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in boot.items()
            if v
        }
        res["by_height"] = {
            f"h{h}": rates(
                [per[s] for s in seqs if s.endswith(f"_h{h}")],
                [area[s] for s in seqs if s.endswith(f"_h{h}")],
            )
            for h in (20, 50, 80)
            if any(s.endswith(f"_h{h}") for s in seqs)
        }
        res["per_sequence"] = {s: rates([per[s]], [area[s]]) for s in seqs}
        result[g] = res
        p = res["pooled"]
        print(
            f"{g}: {p['false_alarms_per_km2']:.1f} false alarms per km2 "
            f"({p['n_false_alarm_tracks']} of {p['n_tracks']} tracks, "
            f"{p['ground_searched_km2']:.3f} km2); found {p['n_truth_found']} of "
            f"{p['n_truth_objects']} ({p['found_share']:.1%})"
        )
    out = REPO_ROOT / "results" / "syndrone" / args.split / f"search_{args.name}.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
