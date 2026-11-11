"""Package HIT-UAV in COCO format for training the thermal detector on a cloud GPU (Phase 13).

Usage: python scripts/hituav_coco_package.py

Reads the HIT-UAV checkout (configs/thermal.yaml, at the pinned commit), maps its
classes onto the VisDrone category numbers (aerialrange.data.hituav.CLASS_MAP; ignored regions
as iscrowd 1) and writes:
  - results/hituav/splits.json: the split by capture day (D-751) and the official split, with
    image and box counts and how far held-out images sit from training frames;
  - cache/gpu_package/hituav_coco.zip and hituav_coco_manifest.json (counts, SHA-256).

Layout inside the zip:
  hituav_coco/images/<file>.jpg                         every image, byte for byte
  hituav_coco/day/{train,valid}/_annotations.coco.json  main split (file names images/<file>)
  hituav_coco/official/{train,valid}/_annotations.coco.json
Test annotations are not packaged: test images are scored only here, once, after the freeze.
Uploaded by hand as a PRIVATE Kaggle dataset; nothing here uploads.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter

from aerialrange.data import hituav as h
from aerialrange.data import visdrone as vd
from aerialrange.utils.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "cache" / "gpu_package"
ROOT = "hituav_coco"
SUB = {"train": "train", "val": "valid"}


def quantiles(v: list[int]) -> dict[str, int] | None:
    if not v:
        return None
    s = sorted(v)
    return {f"q{int(p * 100)}": s[int(p * (len(s) - 1))] for p in (0.1, 0.5, 0.9)}


def summary(images: list[h.Image]) -> dict:
    boxes = Counter(vd.CATEGORIES[b[0]] for im in images for b in im.boxes if not b[5])
    return {
        "images": len(images),
        "night_images": sum(im.night for im in images),
        "altitudes_m": sorted({im.altitude_m for im in images}),
        "angles_deg": sorted({im.angle_deg for im in images}),
        "boxes_per_class": dict(sorted(boxes.items())),
        "ignored_boxes": sum(b[5] for im in images for b in im.boxes),
    }


def main(argv: list[str] | None = None) -> int:
    th = load_config("thermal")
    repo = REPO_ROOT / load_config("data").require("data_root") / th.require("hituav_subdir")
    want = th.require("hituav_commit")
    got = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if got != want:
        print(f"HIT-UAV checkout is at {got}, configs/thermal.yaml pins {want}")
        return 1
    images = h.load(repo)
    days = h.day_split(images)
    day_of = {dd: s for s, v in days.items() for dd in v}
    splits = {
        "day": {im.file: day_of[im.date] for im in images},
        "official": {im.file: im.official_split for im in images},
    }
    record: dict = {
        "dataset": "HIT-UAV",
        "repository_commit": got,
        "rule": "D-751: whole capture days; val and test each hold day and night images; test "
        "holds every altitude level; closest to the paper's 70/10/20 % by image",
        "day_split_days": days,
        "class_map": {
            "Person": "pedestrian (1)",
            "Car": "car (4)",
            "Bicycle": "bicycle (3)",
            "OtherVehicle": "ignored region",
            "DontCare": "ignored region",
        },
    }
    for name, split_of in splits.items():
        part = {}
        for s in h.SPLITS:
            sel = [im for im in images if split_of[im.file] == s]
            part[s] = summary(sel)
            if s != "train":
                part[s]["serial_gap_to_nearest_train_same_setting"] = quantiles(
                    h.nearest_serial_gap(images, split_of, s)
                )
        record[name] = part
        record[name + "_files"] = {
            s: sorted(f for f, v in split_of.items() if v == s) for s in h.SPLITS
        }
    res = REPO_ROOT / "results" / "hituav"
    res.mkdir(parents=True, exist_ok=True)
    (res / "splits.json").write_text(json.dumps(record, indent=1) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    zpath = OUT / "hituav_coco.zip"

    def put(z: zipfile.ZipFile, arcname: str, data: bytes) -> None:
        # a fixed timestamp, so the same data always gives the same zip (and SHA-256)
        info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
        z.writestr(info, data, compress_type=zipfile.ZIP_STORED)

    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for im in images:
            src = repo / "normal_json" / im.official_split / im.file
            put(z, f"{ROOT}/images/{im.file}", src.read_bytes())
        for name, split_of in splits.items():
            for s, sub in SUB.items():
                sel = [im for im in images if split_of[im.file] == s]
                coco = h.coco(sel, file_prefix="images/")
                put(z, f"{ROOT}/{name}/{sub}/_annotations.coco.json", json.dumps(coco).encode())
    sha = hashlib.sha256()
    with zpath.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    counts = {n: {s: record[n][s]["images"] for s in h.SPLITS} for n in splits}
    manifest = {"zip": zpath.name, "sha256": sha.hexdigest(), "images": counts, "commit": got}
    (OUT / "hituav_coco_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("day_split_days", "day", "official")}, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
