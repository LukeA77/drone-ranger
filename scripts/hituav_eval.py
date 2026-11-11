"""Detection AP on HIT-UAV thermal images (Phase 13, D-751).

Usage:
  python scripts/hituav_eval.py --run dfine_hituav_day_coco
         --weights weights/dfine_hituav_day_coco.onnx
         [--split day|official] [--part val|test --allow-test]

Any ONNX detector the project runs (`detect.factory`; D-FINE by default), so the colour
(VisDrone) detector is scored on thermal images the same way as the thermal ones (transfer).
Ground truth from the HIT-UAV checkout with the project's class map (aerialrange.data.hituav):
people = pedestrian, vehicles = car; bicycles are in neither group; ignored regions (DontCare,
OtherVehicle) remove predictions centred inside them, as VisDrone's ignored regions (D-302).
Same protocol as scripts/detect_eval.py: score floor 0.25, IoU iou_match, AP50 and AP50:95
per group, and by box height (version 1's bins), flight altitude, camera angle and time of day
(from the file names). 95 % intervals resample flight settings (time of day, altitude, angle),
since images of one setting come from the same videos. Test images are scored only with
--allow-test, once, after the freeze.
Writes results/thermal/detect_<run>_<split>_<part>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime

import cv2
import numpy as np

from aerialrange.data import hituav as h
from aerialrange.data.visdrone import outside_ignored
from aerialrange.detect.evaluate import ap_from_pieces, ap_pieces
from aerialrange.detect.factory import KINDS, YOLO_PREDICT_CONF, make_detector
from aerialrange.utils.config import REPO_ROOT, load_config

HEIGHT_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]  # scripts/detect_eval.py
COCO_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
N_BOOT = 1000


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--kind", default="dfine", choices=KINDS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="day", choices=["day", "official"])
    ap.add_argument("--part", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    if args.part == "test" and not args.allow_test:
        print("test images are scored once, after the freeze; pass --allow-test")
        return 1
    det_cfg = load_config("detector")
    iou = det_cfg.require("iou_match")
    max_det = det_cfg.require("max_det")
    cmap = {int(k): v for k, v in det_cfg.require("class_maps")["visdrone10"].items()}
    groups = sorted(set(cmap.values()))
    # truth groups by VisDrone category number (the model's class index + 1)
    truth_group = {cid + 1: g for cid, g in cmap.items()}
    th = load_config("thermal")
    repo = REPO_ROOT / load_config("data").require("data_root") / th.require("hituav_subdir")
    splits = json.loads((REPO_ROOT / "results" / "hituav" / "splits.json").read_text())
    files = set(splits[f"{args.split}_files"][args.part])
    images = [im for im in h.load(repo) if im.file in files]
    predict = make_detector(
        args.kind, args.weights, args.imgsz, YOLO_PREDICT_CONF, max_det, sorted(cmap)
    )
    gt = {g: {} for g in groups}
    det = {g: {} for g in groups}
    meta = {}
    t_total = 0.0
    for im in images:
        img = cv2.imread(
            str(repo / "normal_json" / im.official_split / im.file)
        )  # 3 equal channels
        t0 = time.perf_counter()
        b, s, c = predict(img)
        t_total += time.perf_counter() - t0
        ign = np.array([bx[1:5] for bx in im.boxes if bx[5]], float).reshape(-1, 4)
        if len(b) and len(ign):
            keep = outside_ignored(b, ign)
            b, s, c = b[keep], s[keep], c[keep]
        grp = np.array([cmap.get(int(x), "") for x in c])
        for g in groups:
            gt[g][im.file] = np.array(
                [bx[1:5] for bx in im.boxes if not bx[5] and truth_group.get(bx[0]) == g], float
            ).reshape(-1, 4)
            m = grp == g
            det[g][im.file] = (b[m], s[m])
        meta[im.file] = {
            "setting": f"{'night' if im.night else 'day'}_{im.altitude_m}_{im.angle_deg}",
            "time": "night" if im.night else "day",
            "altitude_m": im.altitude_m,
            "angle_deg": im.angle_deg,
        }

    def ap_over(
        g: str, keys: list[str], thr: float = iou, hr: tuple[float, float] | None = None
    ) -> dict:
        sub_gt = {k: gt[g][k] for k in keys}
        sub_det = {k: det[g][k] for k in keys}
        return ap_from_pieces([ap_pieces(sub_gt, sub_det, thr, hr)])

    by_setting = defaultdict(list)
    for k, m in meta.items():
        by_setting[m["setting"]].append(k)
    settings = sorted(by_setting)
    rng = np.random.default_rng(load_config("project").require("seed"))
    res: dict = {}
    for g in groups:
        allk = list(meta)
        pieces = {
            s: ap_pieces(
                {k: gt[g][k] for k in by_setting[s]}, {k: det[g][k] for k in by_setting[s]}, iou
            )
            for s in settings
        }
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(settings, len(settings))
            boots.append(ap_from_pieces([pieces[s] for s in pick])["ap"])
        r = {
            "ap50": ap_over(g, allk),
            "ap50_ci95_resample_settings": [float(np.nanpercentile(boots, q)) for q in (2.5, 97.5)],
            "ap50_95": float(np.mean([ap_over(g, allk, t)["ap"] for t in COCO_THRESHOLDS])),
            "by_box_height_px": {
                f"{lo}-{hi if hi < 1e9 else 'inf'}": ap_over(g, allk, iou, (lo, hi))
                for lo, hi in HEIGHT_BINS
            },
        }
        for key in ("altitude_m", "angle_deg", "time"):
            vals = sorted({m[key] for m in meta.values()})
            r[f"by_{key}"] = {
                str(v): ap_over(g, [k for k, m in meta.items() if m[key] == v]) for v in vals
            }
        res[g] = r
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "weights": args.weights,
        "kind": args.kind,
        "split": args.split,
        "part": args.part,
        "n_images": len(images),
        "n_settings": len(settings),
        "iou_match": iou,
        "score_floor": YOLO_PREDICT_CONF,
        "max_det": max_det,
        "detector_ms_per_image": 1000 * t_total / max(len(images), 1),
        "groups": res,
    }
    path = REPO_ROOT / "results" / "thermal" / f"detect_{args.run}_{args.split}_{args.part}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {
                g: {"ap50": res[g]["ap50"], "ci": res[g]["ap50_ci95_resample_settings"]}
                for g in groups
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
