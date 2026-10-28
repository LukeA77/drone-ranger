"""Test the claim that VisDrone DET images do not overlap the MOT video frames.

Usage: python scripts/visdrone_overlap.py [--frame-stride 5] [--max-hamming 4]

Two checks: (1) file provenance: DET and MOT naming schemes; (2) content: 64-bit average
hashes of every DET train/val image against every `frame-stride`-th frame of every MOT
sequence (all splits), counting pairs within `max-hamming` bits. Neighbouring video frames are
near-identical, so a stride of a few frames still catches a DET image cut from a video.
Writes results/visdrone/overlap.json (counts and the closest pairs, for visual inspection).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from aerialrange.data.visdrone import average_hash
from aerialrange.utils.config import REPO_ROOT, load_config


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--max-hamming", type=int, default=4)
    args = ap.parse_args(argv)
    data = load_config("data")
    root = REPO_ROOT / data.require("data_root")
    det = sorted(
        (root / data.require("visdrone_det_subdir")).glob("VisDrone2019-DET-[tv][ra]*/images/*.jpg")
    )
    mot = []
    for seq_dir in sorted(
        (root / data.require("visdrone_mot_subdir")).glob("VisDrone2019-MOT-*/sequences/*")
    ):
        mot += sorted(seq_dir.glob("*.jpg"))[:: args.frame_stride]
    print(f"{len(det)} DET images, {len(mot)} MOT frames", flush=True)
    det_names = {re.sub(r"_\d+$", "", p.stem.split("_d_")[0]) for p in det}
    mot_names = {p.parent.name for p in mot}
    hd = np.array([average_hash(p) for p in det], dtype=np.uint64)
    print("DET hashed", flush=True)
    hm = np.array([average_hash(p) for p in mot], dtype=np.uint64)
    print("MOT hashed", flush=True)
    best = np.full(len(hd), 64)
    best_j = np.zeros(len(hd), int)
    for i0 in range(0, len(hd), 256):
        d = popcount64(hd[i0 : i0 + 256, None] ^ hm[None, :])
        best[i0 : i0 + 256] = d.min(axis=1)
        best_j[i0 : i0 + 256] = d.argmin(axis=1)
    order = np.argsort(best)[:20]

    # Provenance: both naming schemes carry the source video number
    # (DET "0000201_00182_d_..." and MOT "uav0000201_00000_v").
    def det_vid(p: Path) -> int:
        return int(p.stem.split("_")[0])

    def mot_vid(name: str) -> int:
        return int(name[3:10])

    mot_split_of = {}
    for f in mot:
        mot_split_of.setdefault(mot_vid(f.parent.name), set()).add(
            f.parts[-4].replace("VisDrone2019-MOT-", "")
        )
    prov: dict[str, dict[str, int]] = {}
    for pth in det:
        dsplit = pth.parts[-3].replace("VisDrone2019-DET-", "")
        for ms in sorted(mot_split_of.get(det_vid(pth), {"none"})):
            prov.setdefault(dsplit, {}).setdefault(ms, 0)
            prov[dsplit][ms] += 1
    same_video = np.array(
        [det_vid(det[i]) == mot_vid(mot[best_j[i]].parent.name) for i in range(len(det))]
    )
    close = best <= args.max_hamming
    leak = {}
    for dsplit in ("train", "val"):
        ids = sorted(
            {
                det_vid(pth)
                for pth in det
                if pth.parts[-3].endswith(dsplit)
                and mot_split_of.get(det_vid(pth), set()) & {"val", "test-dev"}
            }
        )
        leak[dsplit] = {
            "video_ids": ids,
            "n_images": sum(
                1 for pth in det if pth.parts[-3].endswith(dsplit) and det_vid(pth) in ids
            ),
        }
    out = {
        "n_det_images": len(det),
        "n_mot_frames_hashed": len(mot),
        "frame_stride": args.frame_stride,
        "max_hamming": args.max_hamming,
        "naming": {
            "det_example": det[0].stem,
            "mot_example_sequence": mot[0].parent.name,
            "det_uses_d_suffix": all("_d_" in p.stem for p in det),
            "mot_uses_v_suffix": all(n.endswith("_v") for n in mot_names),
            "shared_name_stems": len(det_names & mot_names),
        },
        "pairs_within_threshold": int(close.sum()),
        "close_pairs_same_video": int((close & same_video).sum()),
        "close_pairs_different_video": int((close & ~same_video).sum()),
        "det_images_by_source_video_mot_split": prov,
        "det_images_sharing_a_video_with_mot_val_or_testdev": leak,
        "min_hamming_histogram": {
            int(k): int(v) for k, v in zip(*np.unique(best, return_counts=True), strict=True)
        },
        "closest_pairs": [
            {
                "det": str(det[i].relative_to(root)),
                "mot": str(mot[best_j[i]].relative_to(root)),
                "hamming": int(best[i]),
            }
            for i in order
        ],
    }
    dest = REPO_ROOT / "results" / "visdrone" / "overlap.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "closest_pairs"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
