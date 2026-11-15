"""Thermal test images seen by the colour detector and by the thermal detector (README pictures).

Usage: python scripts/thermal_example.py

Images are chosen by written rules, not by eye or by detector score, from the HIT-UAV test days
(the split by capture day, results/hituav/splits.json): the night image with the most labelled
people, and the day image with the most labelled people; ties go to the lower serial number. The
test was already scored once (D-774), so showing its images changes nothing. Both detectors run
on the same image with the same settings; boxes scoring at least SHOW are drawn, people in cyan
and vehicles in white. A person counts as found when a person box overlaps them by half (the
evaluation's rule). The camera records greyscale; the pictures show it in false colour (the
inferno palette) so warm bodies stand out. The night view is cropped to the area holding the
labelled people and enlarged.

Writes docs/media/thermal_night_day.jpg (night above day) and results/thermal/example.json
(images, rules, counts, settings). HIT-UAV is CC BY 4.0.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerialrange.data import hituav as h  # noqa: E402
from aerialrange.detect.factory import YOLO_PREDICT_CONF, make_detector  # noqa: E402
from aerialrange.utils.config import REPO_ROOT, load_config  # noqa: E402

SHOW = 0.5  # draw boxes scoring at least this
DETECTORS = [  # panel title, weights
    ("Colour detector", "weights/dfine_n_640.onnx"),
    ("Thermal detector", "weights/dfine_hituav_day_coco.onnx"),
]
PERSON, VEHICLE = (255, 229, 0), (255, 255, 255)  # BGR: cyan, white
GAP, HEAD, WIDTH = 16, 46, 1100  # gap between views, label strip, output width (px)
PAD = 30  # px around the labelled people in the night crop
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N, 4) and b (M, 4), both left, top, width, height."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = (np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None])).clip(0)
    hh = (np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None])).clip(0)
    inter = w * hh
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def found(truth: np.ndarray, boxes: np.ndarray, thr: float) -> int:
    """People found: one-to-one greedy matching at IoU >= thr."""
    if not len(truth) or not len(boxes):
        return 0
    m = iou(truth, boxes)
    n = 0
    while m.size and m.max() >= thr:
        i, j = np.unravel_index(m.argmax(), m.shape)
        n += 1
        m[i, :] = -1
        m[:, j] = -1
    return n


def font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(FONTS / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
    )


def main() -> int:
    det_cfg = load_config("detector")
    cmap = {int(k): v for k, v in det_cfg.require("class_maps")["visdrone10"].items()}
    truth_group = {cid + 1: g for cid, g in cmap.items()}
    th = load_config("thermal")
    repo = REPO_ROOT / load_config("data").require("data_root") / th.require("hituav_subdir")
    splits = json.loads((REPO_ROOT / "results/hituav/splits.json").read_text(encoding="utf-8"))
    test = set(splits["day_files"]["test"])
    iou_thr = det_cfg.require("iou_match")
    predictors = [
        make_detector(
            "dfine", REPO_ROOT / w, 640, YOLO_PREDICT_CONF, det_cfg.require("max_det"), sorted(cmap)
        )
        for _, w in DETECTORS
    ]

    def labelled(im: h.Image, group: str) -> np.ndarray:
        rows = [b[1:5] for b in im.boxes if not b[5] and truth_group.get(b[0]) == group]
        return np.array(rows, float).reshape(-1, 4)

    images = [im for im in h.load(repo) if im.file in test]

    def pick(night: bool) -> h.Image:
        pool = [im for im in images if im.night == night]
        return max(pool, key=lambda x: (len(labelled(x, "person")), -x.serial))

    def views(im: h.Image, crop: bool) -> tuple[Image.Image, dict]:
        img = cv2.imread(str(repo / "normal_json" / im.official_split / im.file))
        shown = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
        truth = labelled(im, "person")
        out, rec = [], []
        for (name, weights), predict in zip(DETECTORS, predictors, strict=True):
            b, s, c = predict(img)
            grp = np.array([cmap.get(int(x), "") for x in c])
            keep = s >= SHOW
            v = shown.copy()
            for box, g in zip(b[keep], grp[keep], strict=True):
                if g in ("person", "vehicle"):
                    x, y, w, hh = (int(round(q)) for q in box)
                    col = PERSON if g == "person" else VEHICLE
                    cv2.rectangle(v, (x, y), (x + w, y + hh), col, 2)
            pb = b[keep & (grp == "person")]
            rec.append(
                {
                    "detector": name,
                    "weights": weights,
                    "person_boxes": int(len(pb)),
                    "people_found": found(truth, pb, iou_thr),
                    "vehicle_boxes": int((keep & (grp == "vehicle")).sum()),
                }
            )
            out.append(v)
        region = [0, 0, img.shape[1], img.shape[0]]
        if crop:
            x0, y0 = (truth[:, :2].min(0) - PAD).astype(int)
            x1, y1 = (truth[:, :2] + truth[:, 2:]).max(0).astype(int) + PAD
            region = [max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])]
            out = [v[region[1] : region[3], region[0] : region[2]] for v in out]
        half = (WIDTH - GAP) // 2
        out = [
            cv2.resize(
                v, (half, round(v.shape[0] * half / v.shape[1])), interpolation=cv2.INTER_CUBIC
            )
            for v in out
        ]
        hgt = out[0].shape[0]
        sheet = np.full((hgt + HEAD, WIDTH, 3), 255, np.uint8)
        for i, v in enumerate(out):
            sheet[HEAD:, i * (half + GAP) : i * (half + GAP) + half] = v
        pil = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for i, r in enumerate(rec):
            x = i * (half + GAP) + 4
            d.text((x, 12), r["detector"], font=font(True, 19), fill=(20, 20, 20))
            n_p, n_v = r["person_boxes"], r["vehicle_boxes"]
            note = (
                f"{n_p} {'person' if n_p == 1 else 'people'}, "
                f"{n_v} {'vehicle' if n_v == 1 else 'vehicles'} boxed"
            )
            x2 = x + d.textlength(r["detector"], font=font(True, 19)) + 14
            d.text((x2, 14), note, font=font(False, 17), fill=(90, 90, 90))
        info = {
            "image": im.file,
            "night": im.night,
            "altitude_m": im.altitude_m,
            "angle_deg": im.angle_deg,
            "people_labelled": int(len(truth)),
            "vehicles_labelled": int(len(labelled(im, "vehicle"))),
            "region_px": [int(v) for v in region],
            "detectors": rec,
        }
        return pil, info

    night_view, night = views(pick(True), crop=True)
    day_view, day = views(pick(False), crop=False)
    media = REPO_ROOT / "docs/media"

    def titled(view: Image.Image, title: str, sub: str) -> Image.Image:
        c = Image.new("RGB", (WIDTH, view.height + 44), "white")
        d = ImageDraw.Draw(c)
        d.text((4, 10), title, font=font(True, 22), fill=(15, 15, 15))
        x = 14 + d.textlength(title, font=font(True, 22))
        d.text((x, 13), sub, font=font(False, 18), fill=(90, 90, 90))
        c.paste(view, (0, 44))
        return c

    rows = [
        titled(
            night_view,
            f"Night, {night['altitude_m']} m",
            f"{night['people_labelled']} people in view (enlarged)",
        ),
        titled(
            day_view,
            f"Day, {day['altitude_m']} m",
            f"{day['people_labelled']} people and {day['vehicles_labelled']} vehicles in view",
        ),
    ]
    both = Image.new("RGB", (WIDTH, sum(r.height for r in rows) + 20), "white")
    both.paste(rows[0], (0, 0))
    both.paste(rows[1], (0, rows[0].height + 20))
    both.save(media / "thermal_night_day.jpg", quality=88)
    info = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": "from the day-split test days: the night image and the day image with the most "
        "labelled people; ties to the lower serial",
        "show_score": SHOW,
        "iou_match": iou_thr,
        "palette": "greyscale shown in false colour (OpenCV inferno)",
        "night": night,
        "day": day,
        "outputs": ["docs/media/thermal_night_day.jpg"],
        "licence": "HIT-UAV, CC BY 4.0",
    }
    dest = REPO_ROOT / "results/thermal/example.json"
    dest.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("night", "day")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
