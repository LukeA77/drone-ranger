"""SynDrone development package in COCO format, for fine-tuning the version 2 detector (D-740).

Usage: python scripts/syndrone_coco_package.py

Converts cache/gpu_package/syndrone_dev_yolo.zip (scripts/syndrone_package_for_gpu.py, D-707:
every 5th frame, ignored objects filled with the image mean, three leave-one-town-out folds
with the last 20 % of each training town's frames as selection frames) into
cache/gpu_package/syndrone_dev_coco.zip with the same images, so the version 2 detector is
fine-tuned on exactly the data version 1's was. Layout:
  syndrone_dev_coco/images/<Town>/<h>_<frame>.jpg
  syndrone_dev_coco/fold_<Town>/train/_annotations.coco.json   (the fold's training frames)
  syndrone_dev_coco/fold_<Town>/valid/_annotations.coco.json   (the fold's selection frames)
File names in the annotations are relative to syndrone_dev_coco/. Category ids are the
VisDrone numbers 1 to 10 (YOLO id + 1), as in the version 2 detector's VisDrone training
(scripts/visdrone_coco_package.py). Writes cache/gpu_package/syndrone_coco_manifest.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile

from PIL import Image

from aerialrange.utils.config import REPO_ROOT

PKG = REPO_ROOT / "cache" / "gpu_package"
SRC = PKG / "syndrone_dev_yolo.zip"
DST = PKG / "syndrone_dev_coco.zip"
NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def coco_for(zin: zipfile.ZipFile, list_name: str, sizes: dict[str, tuple[int, int]]) -> dict:
    images, anns = [], []
    rels = [
        ln.strip().lstrip("./")
        for ln in zin.read(f"syndrone_dev_yolo/{list_name}").decode().splitlines()
        if ln.strip()
    ]
    for img_id, rel in enumerate(rels, start=1):  # rel: images/<Town>/<name>.jpg
        w, h = sizes[rel]
        images.append({"id": img_id, "file_name": rel, "width": w, "height": h})
        lab = "syndrone_dev_yolo/" + rel.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        for ln in zin.read(lab).decode().splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            bw_px, bh_px = float(bw) * w, float(bh) * h
            x, y = float(cx) * w - bw_px / 2, float(cy) * h - bh_px / 2
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    "bbox": [round(x, 2), round(y, 2), round(bw_px, 2), round(bh_px, 2)],
                    "area": round(bw_px * bh_px, 2),
                    "iscrowd": 0,
                }
            )
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    with zipfile.ZipFile(SRC) as zin:
        names = zin.namelist()
        jpgs = [n for n in names if n.endswith(".jpg")]
        sizes = {}
        for n in jpgs:
            with Image.open(io.BytesIO(zin.read(n))) as im:
                sizes[n.removeprefix("syndrone_dev_yolo/")] = im.size
        folds = sorted(
            n.split("fold_")[1].removesuffix("_train.txt")
            for n in names
            if n.endswith("_train.txt")
        )
        counts = {}
        with zipfile.ZipFile(DST, "w", zipfile.ZIP_STORED) as zout:
            for n in jpgs:  # JPEGs are already compressed
                zout.writestr(n.replace("syndrone_dev_yolo/", "syndrone_dev_coco/", 1), zin.read(n))
            for town in folds:
                counts[town] = {}
                for split, lst in (("train", "train"), ("valid", "select")):
                    coco = coco_for(zin, f"fold_{town}_{lst}.txt", sizes)
                    zout.writestr(
                        f"syndrone_dev_coco/fold_{town}/{split}/_annotations.coco.json",
                        json.dumps(coco),
                    )
                    counts[town][split] = {
                        "images": len(coco["images"]),
                        "boxes": len(coco["annotations"]),
                    }
    sha = hashlib.sha256(DST.read_bytes()).hexdigest()
    manifest = {
        "decision": "D-740",
        "source": SRC.name,
        "zip": DST.name,
        "sha256": sha,
        "folds": counts,
        "category_ids": {i + 1: n for i, n in enumerate(NAMES)},
    }
    (PKG / "syndrone_coco_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
