"""Prepare VisDrone for Phase 3.

Usage: python scripts/visdrone_prep.py [--mot] [--det]

--mot: for each MOT split, write MOTChallenge ground truth per evaluation group to
       cache/visdrone/mot/<split>/<sequence>/gt_<group>.txt, plus the ignored regions and
       seqinfo (frame count and image size), and a summary in results/visdrone/mot_prep.json.
--det: build a YOLO-format detection set from VisDrone DET train/val under
       cache/visdrone/yolo/ (images linked, labels written) and a dataset YAML.
Data are read from the extracted folders under the data root; nothing is modified there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from aerialrange.data import visdrone as vd
from aerialrange.utils.config import REPO_ROOT, load_config

CACHE = REPO_ROOT / "cache" / "visdrone"
MOT_SPLITS = {
    "train": "VisDrone2019-MOT-train",
    "val": "VisDrone2019-MOT-val",
    "test-dev": "VisDrone2019-MOT-test-dev",
}
DET_SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}


def prep_mot(root: Path, groups: dict[str, list[int]]) -> dict:
    summary = {}
    for split, folder in MOT_SPLITS.items():
        base = root / folder
        if not base.is_dir():
            summary[split] = "missing (extracted folder not found)"
            continue
        seqs = sorted(p.stem for p in (base / "annotations").glob("*.txt"))
        counts: Counter = Counter()
        for seq in seqs:
            a = vd.parse_mot((base / "annotations" / f"{seq}.txt").read_text())
            out = CACHE / "mot" / split / seq
            out.mkdir(parents=True, exist_ok=True)
            for g in groups:
                lines = vd.to_motchallenge_gt(a, groups, g)
                (out / f"gt_{g}.txt").write_text("\n".join(lines) + "\n")
                counts[g] += len(lines)
            ign = a[a["category"] == vd.IGNORED]
            (out / "ignored.txt").write_text(
                "\n".join(
                    f"{r['frame']},{int(r['left'])},{int(r['top'])},{int(r['width'])},"
                    f"{int(r['height'])}"
                    for r in ign
                )
                + "\n"
            )
            frames = sorted((base / "sequences" / seq).glob("*.jpg"))
            w, h = Image.open(frames[0]).size
            (out / "seqinfo.json").write_text(
                json.dumps(
                    {
                        "sequence": seq,
                        "frames": len(frames),
                        "width": w,
                        "height": h,
                        "frame_dir": str((base / "sequences" / seq).relative_to(REPO_ROOT)),
                    }
                )
                + "\n"
            )
            counts["ignored_regions"] += len(ign)
        summary[split] = {"sequences": len(seqs), **counts}
        print(split, summary[split], flush=True)
    return summary


def excluded_videos() -> dict[str, set[int]]:
    """DET source videos shared with MOT val/test-dev (results/visdrone/overlap.json, D-308).
    Their images are left out of detector training and model selection to avoid leakage."""
    f = REPO_ROOT / "results" / "visdrone" / "overlap.json"
    if not f.exists():
        raise FileNotFoundError("run scripts/visdrone_overlap.py first (leakage check)")
    r = json.loads(f.read_text())["det_images_sharing_a_video_with_mot_val_or_testdev"]
    return {s: set(v["video_ids"]) for s, v in r.items()}


def prep_det(root: Path) -> dict:
    summary = {}
    excl = excluded_videos()
    for split, folder in DET_SPLITS.items():
        base = root / folder
        if not base.is_dir():
            summary[split] = "missing (extracted folder not found)"
            continue
        img_out = CACHE / "yolo" / "images" / split
        lab_out = CACHE / "yolo" / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        n_img = n_box = n_excl = 0
        for stale in list(img_out.glob("*.jpg")) + list(lab_out.glob("*.txt")):
            stale.unlink()  # rebuild from scratch so exclusions always apply
        for img in sorted((base / "images").glob("*.jpg")):
            if int(img.stem.split("_")[0]) in excl.get(split, set()):
                n_excl += 1
                continue
            w, h = Image.open(img).size
            lines = vd.det_to_yolo((base / "annotations" / f"{img.stem}.txt").read_text(), w, h)
            (lab_out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            link = img_out / img.name
            if not link.exists():
                try:
                    os.link(img, link)  # hard link: no copy, same volume
                except OSError:
                    link.write_bytes(img.read_bytes())
            n_img += 1
            n_box += len(lines)
        summary[split] = {
            "images": n_img,
            "boxes": n_box,
            "excluded_images_shared_video_with_mot_eval": n_excl,
        }
        print(split, summary[split], flush=True)
    names = [vd.CATEGORIES[c] for c in vd.TRAIN_CLASSES]
    (CACHE / "yolo" / "visdrone.yaml").write_text(
        f"path: {(CACHE / 'yolo').as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names: {json.dumps(names)}\n"
    )  # YAML flow list: index = class id
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot", action="store_true")
    ap.add_argument("--det", action="store_true")
    args = ap.parse_args(argv)
    data = load_config("data")
    root = REPO_ROOT / data.require("data_root")
    out = {}
    if args.mot:
        out["mot"] = prep_mot(
            root / data.require("visdrone_mot_subdir"), data.require("visdrone_eval_groups")
        )
    if args.det:
        out["det"] = prep_det(root / data.require("visdrone_det_subdir"))
    dest = REPO_ROOT / "results" / "visdrone" / "prep.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(dest.read_text()) if dest.exists() else {}
    old.update(out)
    dest.write_text(json.dumps(old, indent=2) + "\n")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
