"""Package VisDrone DET train/val in COCO format for training detector candidates on a cloud GPU.

Usage: python scripts/visdrone_coco_package.py

Same images and labels as the version 1 detector's training set (scripts/visdrone_prep.py
--det): the 10 VisDrone classes; ignored regions and "others" dropped, images left as they are
(D-707); DET images sharing a source video with MOT val or test-dev left out (D-308). Only the
format differs: COCO JSON, which the candidates' training code reads.

Layout inside the zip (the folder names the candidates' loaders expect):
  visdrone_det_coco/train/_annotations.coco.json + images
  visdrone_det_coco/valid/_annotations.coco.json + images
Images are stored byte for byte (no re-encoding). COCO boxes are [x, y, width, height] in
pixels; category ids 1..10 are the VisDrone category numbers; `area` is the box area;
`iscrowd` 0.

Writes cache/gpu_package/visdrone_det_coco.zip and visdrone_coco_manifest.json (counts and
the zip's SHA-256). Uploaded by hand as a PRIVATE Kaggle dataset; nothing here uploads.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile

from PIL import Image

from aerialrange.data import visdrone as vd
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "cache" / "gpu_package"
SPLITS = {"train": ("VisDrone2019-DET-train", "train"), "val": ("VisDrone2019-DET-val", "valid")}
ROOT = "visdrone_det_coco"


def excluded_videos() -> dict[str, set[int]]:
    """DET source videos shared with MOT val/test-dev (results/visdrone/overlap.json, D-308)."""
    r = json.loads((REPO_ROOT / "results" / "visdrone" / "overlap.json").read_text())
    return {
        s: set(v["video_ids"])
        for s, v in r["det_images_sharing_a_video_with_mot_val_or_testdev"].items()
    }


def coco_for_image(text: str, image_id: int, first_ann_id: int) -> list[dict]:
    """COCO annotations for one DET image, with version 1's rules (`vd.det_to_yolo`)."""
    anns = []
    for r in vd.parse_det(text):
        c = int(r["category"])
        if c not in vd.TRAIN_CLASSES or r["width"] <= 0 or r["height"] <= 0:
            continue
        w, h = float(r["width"]), float(r["height"])
        anns.append(
            {
                "id": first_ann_id + len(anns),
                "image_id": image_id,
                "category_id": c,
                "bbox": [float(r["left"]), float(r["top"]), w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )
    return anns


def main(argv: list[str] | None = None) -> int:
    d = load_config("data")
    data_root = REPO_ROOT / d.require("data_root")
    excl = excluded_videos()
    cats = [
        {"id": c, "name": vd.CATEGORIES[c], "supercategory": "object"} for c in vd.TRAIN_CLASSES
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = OUT / "visdrone_det_coco.zip"
    counts: dict = {}
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for split, (folder, sub) in SPLITS.items():
            base = data_root / "VisDrone-DET" / folder
            if not base.is_dir():
                base = data_root / folder
            if not (base / "images").is_dir():
                print(f"missing {folder} under {data_root}")
                return 1
            images, anns = [], []
            n_excl = 0
            for img in sorted((base / "images").glob("*.jpg")):
                if int(img.stem.split("_")[0]) in excl.get(split, set()):
                    n_excl += 1
                    continue
                with Image.open(img) as im:
                    w, h = im.size
                image_id = len(images) + 1
                images.append({"id": image_id, "file_name": img.name, "width": w, "height": h})
                text = (base / "annotations" / f"{img.stem}.txt").read_text()
                anns += coco_for_image(text, image_id, len(anns) + 1)
                z.write(img, f"{ROOT}/{sub}/{img.name}")
            coco = {"images": images, "annotations": anns, "categories": cats}
            z.writestr(f"{ROOT}/{sub}/_annotations.coco.json", json.dumps(coco))
            per_class = {vd.CATEGORIES[c["id"]]: 0 for c in cats}
            for a in anns:
                per_class[vd.CATEGORIES[a["category_id"]]] += 1
            counts[split] = {
                "images": len(images),
                "boxes": len(anns),
                "boxes_per_class": per_class,
                "excluded_images_shared_video_with_mot_eval": n_excl,
            }
            print(split, counts[split], flush=True)
    h = hashlib.sha256()
    with zpath.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    manifest = {"zip": zpath.name, "sha256": h.hexdigest(), "counts": counts}
    (OUT / "visdrone_coco_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
