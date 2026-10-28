"""Package the prepared VisDrone YOLO set for training on a cloud GPU (Kaggle or Colab).

Usage: python scripts/package_for_gpu.py

Writes cache/gpu_package/visdrone_yolo.zip containing images/, labels/ and a dataset YAML
with a relative path, plus cache/gpu_package/manifest.json (image and label counts, the
D-308 exclusions, and the zip's SHA-256 so the upload can be checked). The zip is uploaded by
hand as a PRIVATE dataset (VisDrone has no stated licence); nothing here uploads anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile

from aerialrange.data.visdrone import CATEGORIES, TRAIN_CLASSES
from aerialrange.utils.config import REPO_ROOT

SRC = REPO_ROOT / "cache" / "visdrone" / "yolo"
OUT = REPO_ROOT / "cache" / "gpu_package"


def main() -> int:
    if not (SRC / "images" / "train").is_dir():
        print("run scripts/visdrone_prep.py --det first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = OUT / "visdrone_yolo.zip"
    names = [CATEGORIES[c] for c in TRAIN_CLASSES]
    yaml_text = (
        "# path is set at run time by the training notebook\n"
        "train: images/train\nval: images/val\n"
        f"names: {json.dumps(names)}\n"
    )
    counts = {}
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("visdrone_yolo/visdrone.yaml", yaml_text)
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                files = sorted((SRC / kind / split).iterdir())
                counts[f"{kind}/{split}"] = len(files)
                for f in files:
                    z.write(f, f"visdrone_yolo/{kind}/{split}/{f.name}")
    h = hashlib.sha256()
    with zpath.open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    prep = json.loads((REPO_ROOT / "results" / "visdrone" / "prep.json").read_text())
    manifest = {
        "zip": zpath.name,
        "bytes": zpath.stat().st_size,
        "sha256": h.hexdigest(),
        "counts": counts,
        "excluded_for_leakage": {
            s: v.get("excluded_images_shared_video_with_mot_eval")
            for s, v in prep.get("det", {}).items()
        },
        "class_names": names,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
