"""Fit the box-to-pixel noise model {{SIGMA_PIX_MODEL}}.

Usage: python scripts/fit_pixel_noise.py --weights cache/train/B2_yolo26n_640/weights/best.pt
       --class-map visdrone10 --imgsz 640 [--frame-stride 10]

Detections on VisDrone MOT val (every `frame-stride`-th frame) are matched one-to-one to ground
truth at IoU >= `{{IOU_MATCH}}` per evaluation group. For each match, the error of the
bottom-centre contact point is divided by the ground-truth box size:
  u error / box width,  v error / box height.
kappa_u, kappa_v are robust standard deviations (1.4826 x MAD) of those ratios, so
sigma_u = kappa_u * width, sigma_v = kappa_v * height. Reported per group, per
box-height bin (to check that "proportional to size" holds) and overall. Validation split only.
Writes results/visdrone/val/<name>.json (default pixel_noise.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import cv2
import numpy as np

from aerialrange.data.visdrone import outside_ignored
from aerialrange.detect.evaluate import match_image
from aerialrange.track.evaluate import read_ignored, read_rows
from aerialrange.utils.config import REPO_ROOT, load_config

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"
HEIGHT_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]  # as D-312


def robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--name", default="pixel_noise", help="output file stem")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    det_cfg = load_config("detector")
    iou = det_cfg.require("iou_match")
    cmap = {int(k): v for k, v in det_cfg.require("class_maps")[args.class_map].items()}
    groups = sorted(set(cmap.values()))
    model = YOLO(REPO_ROOT / args.weights)
    rows: dict[str, list] = {g: [] for g in groups}
    for seq in sorted(p for p in PREP.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        ign = read_ignored(seq / "ignored.txt")
        gts = {g: read_rows(seq / f"gt_{g}.txt", 6) for g in groups}
        for k in range(0, len(frames), args.frame_stride):
            fnum = k + 1
            r = model.predict(
                cv2.imread(str(frames[k])),
                imgsz=args.imgsz,
                classes=sorted(cmap),
                max_det=det_cfg.require("max_det"),
                verbose=False,
            )[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            b = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            if fnum in ign and len(b):
                keep = outside_ignored(b, ign[fnum])
                b, s, c = b[keep], s[keep], c[keep]
            grp = np.array([cmap.get(int(x), "") for x in c])
            for g in groups:
                gt = gts[g][gts[g][:, 0] == fnum][:, 2:6]
                m = grp == g
                if not len(gt) or not m.any():
                    continue
                tp, match = match_image(gt, b[m], s[m], iou)
                for j in np.nonzero(tp)[0]:
                    d, t = b[m][j], gt[match[j]]
                    du = (d[0] + d[2] / 2) - (t[0] + t[2] / 2)
                    dv = (d[1] + d[3]) - (t[1] + t[3])
                    rows[g].append((du / t[2], dv / t[3], t[3]))
        print(f"{seq.name} done", flush=True)

    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "weights": args.weights,
        "imgsz": args.imgsz,
        "frame_stride": args.frame_stride,
        "iou_match": iou,
        "contact_point": "bottom_centre",
        "groups": {},
    }
    allr = []
    for g, lst in rows.items():
        a = np.array(lst).reshape(-1, 3)
        allr.append(a)
        by_h = {}
        for lo, hi in HEIGHT_BINS:
            m = (a[:, 2] >= lo) & (a[:, 2] < hi)
            by_h[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = {
                "n": int(m.sum()),
                "kappa_u": robust_sigma(a[m, 0]),
                "kappa_v": robust_sigma(a[m, 1]),
                "bias_v": float(np.median(a[m, 1])) if m.any() else float("nan"),
            }
        out["groups"][g] = {
            "n": len(a),
            "kappa_u": robust_sigma(a[:, 0]),
            "kappa_v": robust_sigma(a[:, 1]),
            "bias_u": float(np.median(a[:, 0])) if len(a) else float("nan"),
            "bias_v": float(np.median(a[:, 1])) if len(a) else float("nan"),
            "by_height": by_h,
        }
    a = np.concatenate(allr)
    out["overall"] = {
        "n": len(a),
        "kappa_u": robust_sigma(a[:, 0]),
        "kappa_v": robust_sigma(a[:, 1]),
    }
    dest = REPO_ROOT / "results" / "visdrone" / "val" / f"{args.name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {g: {k: v for k, v in d.items() if k != "by_height"} for g, d in out["groups"].items()},
            indent=1,
        )
    )
    print("overall", out["overall"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
