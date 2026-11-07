"""Check the project's YOLO runner (ONNX Runtime alone) against the library on the same model.

Usage: python scripts/check_yolo_onnx.py --onnx weights/B2_yolo26n_960_rect.onnx --imgsz 960
       [--score-floor 0.1] [--n-frames 50]

Runs the version 1 detector's ONNX export through Ultralytics' predict and through
`aerialrange.detect.onnx_detectors.YoloOnnx` on the seeded sample of VisDrone MOT val frames
used by the other export checks, and reports the largest box and score differences between
matched detections and the share matched both ways (IoU >= 0.5, same class). Both sides run
the same ONNX file, so any difference is pre- or post-processing. This is the evidence for
running the version 1 detector without its AGPL library. Writes
results/edge/parity_own_<onnx stem>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from aerialrange.detect.onnx_detectors import YoloOnnx
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_rfdetr import compare  # noqa: E402

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--score-floor", type=float, default=0.1)
    ap.add_argument("--n-frames", type=int, default=50)
    args = ap.parse_args(argv)
    import cv2
    from ultralytics import YOLO

    max_det = int(load_config("detector").require("max_det"))
    lib = YOLO(str(REPO_ROOT / args.onnx), task="detect")
    own = YoloOnnx(REPO_ROOT / args.onnx, args.score_floor, max_det)
    frames = []
    for seq in sorted(p for p in PREP.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames += sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
    rng = np.random.default_rng(load_config("project").require("seed"))
    pick = sorted(rng.choice(len(frames), size=min(args.n_frames, len(frames)), replace=False))
    rows = []
    for k in pick:
        img = cv2.imread(str(frames[k]))
        r = lib.predict(
            img, imgsz=args.imgsz, conf=args.score_floor, max_det=max_det, verbose=False
        )[0]
        a = (
            r.boxes.xyxy.cpu().numpy().astype(float),
            r.boxes.conf.cpu().numpy().astype(float),
            r.boxes.cls.cpu().numpy().astype(int),
        )
        d = own(img)
        rows.append(compare(a, (d.xyxy, d.score, d.cls)))
    tot_a = sum(r["n_a"] for r in rows)
    tot_b = sum(r["n_b"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    diffs = sum((r["box_diffs"] for r in rows), [])
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "onnx": Path(args.onnx).name,
        "input_hw": [own.h, own.w],
        "score_floor": args.score_floor,
        "n_frames": len(pick),
        "ultralytics": version("ultralytics"),
        "onnxruntime": version("onnxruntime"),
        "max_box_diff_px": max(r["max_box_diff_px"] for r in rows),
        "p99_box_diff_px": float(np.percentile(diffs, 99)) if diffs else 0.0,
        "max_score_diff": max(r["max_score_diff"] for r in rows),
        "agreement_library_side": matched / max(tot_a, 1),
        "agreement_own_side": matched / max(tot_b, 1),
        "detections_library": tot_a,
        "detections_own": tot_b,
    }
    dest = REPO_ROOT / "results" / "edge" / f"parity_own_{Path(args.onnx).stem}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
