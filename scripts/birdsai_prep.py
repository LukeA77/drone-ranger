"""Convert BIRDSAI real videos to the project's MOT layout (Phase 13 thermal tracking, D-754).

Usage: python scripts/birdsai_prep.py

Reads Data/birdsai/{TrainReal,TestReal}/ (LILA "Conservation Drones", CDLA-Permissive) and
writes cache/birdsai/mot/{train,test}/<video>/ like scripts/visdrone_prep.py --mot does:
seqinfo.json (frames, size, frame_dir), gt_person.txt (MOT rows frame, id, left, top, width,
height, 1, 1, 1; humans only), gt_vehicle.txt (empty: BIRDSAI has no vehicles) and
ignored.txt (empty: BIRDSAI marks no ignored regions). Animals are not ignored: a person
detection on an animal is a false alarm, as it would be for a search crew.

Frame numbering: an annotation's frame k is the k-th image of the video in file-name order
(0-based), not the image whose file number is k. Checked on video 0000000011 (test), whose
files start at 87: under this reading boxes are brighter than their surroundings by 12 grey
levels on average, under the other by -1.7. MOT rows here are 1-based (k + 1).
"""

from __future__ import annotations

import csv
import json
import sys

import cv2

from aerialrange.utils.config import REPO_ROOT, load_config

PARTS = {"TrainReal": "train", "TestReal": "test"}
HUMAN = "1"


def main(argv: list[str] | None = None) -> int:
    d = load_config("data")
    root = REPO_ROOT / d.require("data_root") / "birdsai"
    out_root = REPO_ROOT / "cache" / "birdsai" / "mot"
    summary = {}
    for part, split in PARTS.items():
        n_vid = n_box = 0
        for ann in sorted((root / part / "annotations").glob("*.csv")):
            vid = ann.stem
            img_dir = root / part / "images" / vid
            imgs = sorted(img_dir.glob("*.jpg"))
            if not imgs:
                print(f"{vid}: no images")
                return 1
            h, w = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE).shape
            rows = []
            for r in csv.reader(ann.open()):
                if not r or r[6] != HUMAN:
                    continue
                k = int(r[0])
                if not 0 <= k < len(imgs):
                    print(f"{vid}: annotation frame {k} beyond {len(imgs)} images")
                    return 1
                x, y, bw, bh = (float(v) for v in r[2:6])
                if bw <= 0 or bh <= 0:
                    continue
                rows.append(f"{k + 1},{int(r[1])},{x:g},{y:g},{bw:g},{bh:g},1,1,1")
            seq = out_root / split / vid
            seq.mkdir(parents=True, exist_ok=True)
            (seq / "gt_person.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            (seq / "gt_vehicle.txt").write_text("")
            (seq / "ignored.txt").write_text("")
            info = {
                "sequence": vid,
                "frames": len(imgs),
                "width": w,
                "height": h,
                "frame_dir": str(img_dir.relative_to(REPO_ROOT)),
            }
            (seq / "seqinfo.json").write_text(json.dumps(info))
            n_vid += 1
            n_box += len(rows)
        summary[split] = {"videos": n_vid, "person_boxes": n_box}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
