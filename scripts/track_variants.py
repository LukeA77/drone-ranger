"""Score named tracker variants on VisDrone MOT validation, from cached detections.

Usage:
  python scripts/track_variants.py --name parity --dets B2_yolo26n_960 \
      --variant own_ds2 configs/trackers/motion_parity.yaml '{}' \
      --variant own_ds8 configs/trackers/motion_parity.yaml '{"gmc_downscale": 8}' \
      --variant v1_ds8 configs/trackers/botsort_cmc_tuned_B2_yolo26n_960.yaml '{"gmc_downscale": 8}'

Each variant is (label, tracker YAML, JSON overrides) and is run with the same replay and
scoring as the version 1 tracker grid (`scripts/track_grid.py`, `run_point`): every validation
sequence, person and vehicle, IoU match from configs/detector.yaml. Variants run in
parallel. Validation only. Output: results/visdrone/val/track_variants_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_grid import run_point  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    from aerialrange.utils.config import REPO_ROOT, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dets", default="B2_yolo26n_960")
    ap.add_argument("--class-map", default="visdrone10")
    ap.add_argument(
        "--variant", nargs=3, action="append", required=True, metavar=("LABEL", "YAML", "OVERRIDES")
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    det = load_config("detector")
    iou = det.require("iou_match")
    cmap = {int(k): v for k, v in det.require("class_maps")[args.class_map].items()}
    variants = [(lab, yml, json.loads(ov)) for lab, yml, ov in args.variant]
    with ProcessPoolExecutor(min(args.workers, len(variants))) as ex:
        futs = {
            lab: ex.submit(run_point, args.dets, yml, ov, cmap, iou) for lab, yml, ov in variants
        }
        res = {}
        for lab, yml, _ in variants:
            r = futs[lab].result()
            o = {g: r["metrics"][g]["OVERALL"] for g in r["metrics"]}
            print(
                f"{lab}: mean IDF1 {r['mean_idf1']:.4f} mean MOTA {r['mean_mota']:.4f} "
                + " ".join(
                    f"{g} IDF1 {o[g]['idf1']:.4f} IDsw {int(o[g]['num_switches'])}" for g in o
                )
                + f" ({r['tracker_seconds']:.0f} s)",
                flush=True,
            )
            res[lab] = {"tracker": yml, **r}
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "val",
        "dets": args.dets,
        "iou_match": iou,
        "variants": res,
    }
    dest = REPO_ROOT / "results" / "visdrone" / "val" / f"track_variants_{args.name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
