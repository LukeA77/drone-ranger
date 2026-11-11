"""HIT-UAV thermal dataset (Phase 13): reading, class mapping and the split by capture day.

HIT-UAV (Suo et al., Scientific Data 10, 227, 2023; CC BY 4.0, D-748) holds 2,898 thermal
images (640 x 512, 8-bit, one channel) from a DJI Zenmuse XT2 (FLIR longwave sensor, 25 mm
lens). File names are T_HHH_AA_W_NNNNN: T 0 day, 1 night; HHH flight altitude in metres (60 to
130 in steps of 10); AA camera angle in degrees (30 to 90); W weather (0, no rain); NNNNN a
serial number. Serial numbers are unique and run through the whole dataset in capture order.

The official split (70 / 10 / 20 % by image, the paper's hold-out method) puts neighbouring
frames of one video in different splits: most test images are 1 to 4 serials from a training
image of the same setting. So the project's main split keeps whole capture days apart
(D-751). The official split is kept for comparison with published results.

Classes map onto the VisDrone category numbers the version 2 detector already uses, so the
thermal detector has the same output head and the colour and thermal detectors can be scored on
each other's data with the same groups (configs/data.yaml visdrone_eval_groups).
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# HIT-UAV category id -> (VisDrone category id, ignore). Person -> pedestrian, Car -> car,
# Bicycle -> bicycle (outside both evaluation groups, as in VisDrone). DontCare marks objects
# the annotators could not identify, for example possible people (paper, Fig. 5c), so it is an
# ignored region. OtherVehicle is not defined further in the paper or repository, so it cannot
# be put in the vehicle group (car, van, truck, bus) with confidence; it is ignored, like
# VisDrone's "others".
CLASS_MAP: dict[int, tuple[int, bool]] = {
    0: (1, False),  # Person -> pedestrian
    1: (4, False),  # Car -> car
    2: (3, False),  # Bicycle -> bicycle
    3: (4, True),  # OtherVehicle -> ignored region
    4: (1, True),  # DontCare -> ignored region
}
SPLITS = ("train", "val", "test")
ALTITUDES_M = tuple(range(60, 131, 10))
PAPER_SHARES = {"train": 0.7, "val": 0.1, "test": 0.2}


@dataclass(frozen=True)
class Image:
    file: str
    official_split: str
    date: str
    night: bool
    altitude_m: int
    angle_deg: int
    serial: int
    width: int
    height: int
    # (VisDrone id, x, y, w, h, ignored)
    boxes: tuple[tuple[int, float, float, float, float, bool], ...]


def parse_name(name: str) -> tuple[bool, int, int, int, int]:
    """(night, altitude m, angle deg, weather, serial) from T_HHH_AA_W_NNNNN.jpg."""
    t, alt, ang, w, n = Path(name).stem.split("_")
    if t not in ("0", "1"):
        raise ValueError(f"{name}: time field must be 0 (day) or 1 (night)")
    return t == "1", int(alt), int(ang), int(w), int(n)


def load(repo: Path) -> list[Image]:
    """All images of a HIT-UAV checkout (normal_json annotations: standard boxes)."""
    out = []
    for split in SPLITS:
        d = json.loads((repo / "normal_json" / "annotations" / f"{split}.json").read_text())
        boxes = defaultdict(list)
        for a in d["annotation"]:
            cid, ign = CLASS_MAP[a["category_id"]]
            x, y, w, h = (float(v) for v in a["bbox"])
            if w <= 0 or h <= 0:  # as version 1's VisDrone rule
                continue
            boxes[a["image_id"]].append((cid, x, y, w, h, ign))
        for i in d["images"]:
            night, alt, ang, _, serial = parse_name(i["filename"])
            out.append(
                Image(
                    file=i["filename"],
                    official_split=split,
                    date=i["date_captured"],
                    night=night,
                    altitude_m=alt,
                    angle_deg=ang,
                    serial=serial,
                    width=i["width"],
                    height=i["height"],
                    boxes=tuple(boxes[i["id"]]),
                )
            )
    out.sort(key=lambda im: im.serial)
    return out


def day_split(images: list[Image]) -> dict[str, list[str]]:
    """Capture days for train / val / test (D-751).

    Rule: whole days only; validation and test each hold day and night images; test holds every
    altitude level present in the data, so results by altitude cover the full range; among the
    splits that qualify, the one closest to the paper's 70 / 10 / 20 % (sum of absolute
    differences in image share for val and test), ties broken by the sorted day lists.
    """
    by_day: dict[str, list[Image]] = defaultdict(list)
    for im in images:
        by_day[im.date].append(im)
    days = sorted(by_day)
    alts = {im.altitude_m for im in images}
    n = len(images)
    best = None
    for code in itertools.product(range(3), repeat=len(days)):
        parts = {
            s: [d for d, c in zip(days, code, strict=True) if c == k] for k, s in enumerate(SPLITS)
        }
        if not parts["val"] or not parts["test"] or not parts["train"]:
            continue
        ims = {s: [im for d in parts[s] for im in by_day[d]] for s in SPLITS}
        if any({im.night for im in ims[s]} != {True, False} for s in ("val", "test")):
            continue
        if {im.altitude_m for im in ims["test"]} != alts:
            continue
        err = sum(abs(len(ims[s]) / n - PAPER_SHARES[s]) for s in ("val", "test"))
        key = (round(err, 12), parts["val"], parts["test"])
        if best is None or key < best[0]:
            best = (key, parts)
    if best is None:
        raise ValueError("no split by capture day meets the rule")
    return best[1]


def nearest_serial_gap(images: list[Image], split_of: dict[str, str], held: str) -> list[int]:
    """For each image in split `held`, the serial distance to the nearest training image of the
    same setting (day or night, altitude, angle); a measure of how close held-out images sit to
    training frames of the same video. Images with no training image of their setting are left
    out."""
    train = defaultdict(list)
    for im in images:
        if split_of[im.file] == "train":
            train[(im.night, im.altitude_m, im.angle_deg)].append(im.serial)
    gaps = []
    for im in images:
        if split_of[im.file] != held:
            continue
        cand = train.get((im.night, im.altitude_m, im.angle_deg))
        if cand:
            gaps.append(min(abs(s - im.serial) for s in cand))
    return gaps


def coco(images: list[Image], file_prefix: str = "") -> dict:
    """COCO detection JSON. Ignored regions are written as iscrowd 1: D-FINE's loader drops
    them from training, and COCO evaluation neither rewards nor penalises detections on them
    (within the same category; the project's own scoring drops predictions on any ignored
    region, as for VisDrone)."""
    from aerialrange.data import visdrone as vd

    cats = [{"id": i, "name": vd.CATEGORIES[i]} for i in vd.TRAIN_CLASSES]
    imgs, anns = [], []
    for k, im in enumerate(images, start=1):
        imgs.append(
            {"id": k, "file_name": file_prefix + im.file, "width": im.width, "height": im.height}
        )
        for cid, x, y, w, h, ign in im.boxes:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": k,
                    "category_id": cid,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": int(ign),
                }
            )
    return {"images": imgs, "annotations": anns, "categories": cats}
