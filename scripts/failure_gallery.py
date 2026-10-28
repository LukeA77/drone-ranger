"""Failure gallery on MOT val.

Usage: python scripts/failure_gallery.py --run B2_yolo26n_960_tuned [--top 12]

Reads the tracks and events written by scripts/track_tuned.py. Two galleries per group:
1. Identity switches: the switch events of the sequences with the most switches. Each
   thumbnail shows the frame before and the frame of the switch, ground truth in green and the
   tracker's boxes in red.
2. Most-missed small objects: ground-truth objects with median box height below 32 px (the
   upper edge of the 16-32 px size bin used throughout) that were missed in the largest share
   of their frames (at least 20 frames).
Each item gets measured tags, not guessed causes: box height, VisDrone occlusion and truncation
levels (raw annotation columns; occlusion 1 = partial, 2 = heavy), and the camera shift between
consecutive frames (phase correlation on quarter-size frames, in full-size pixels), tagged when
it exceeds the box height, beyond which an uncompensated box no longer overlaps its next
position. The one-line cause names the tags that apply.
Writes results/failures/<run>/ (JPEG sheets and CSVs).
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd

from aerialrange.utils.config import REPO_ROOT

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"
SMALL_PX = 32
THUMB = 200


def frame_paths(seq: str) -> list:
    info = json.loads((PREP / seq / "seqinfo.json").read_text())
    return sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))


def raw_annotations(seq: str) -> pd.DataFrame:
    p = REPO_ROOT / "Data" / "VisDrone-MOT" / "VisDrone2019-MOT-val" / "annotations" / f"{seq}.txt"
    cols = ["frame", "id", "l", "t", "w", "h", "score", "cat", "trunc", "occl"]
    return pd.read_csv(p, header=None, names=cols)


def camera_shift(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.25
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s).astype(np.float32)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return float(np.hypot(dx, dy) / s)


def crop(img: np.ndarray, box: np.ndarray, gt: list, pred: list, label: str) -> np.ndarray:
    x, y, w, h = box
    cx, cy, half = x + w / 2, y + h / 2, max(48.0, 1.5 * max(w, h))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    im = img.copy()
    for bx, col in [(b, (0, 200, 0)) for b in gt] + [(b, (0, 0, 255)) for b in pred]:
        bx0, by0, bw, bh = (int(v) for v in bx)
        cv2.rectangle(im, (bx0, by0), (bx0 + bw, by0 + bh), col, 1)
    c = cv2.resize(im[y0:y1, x0:x1], (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)
    cv2.putText(c, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return c


def boxes_at(rows: np.ndarray, f: int) -> list:
    return [r[2:6] for r in rows[rows[:, 0] == f]] if len(rows) else []


def cause(h: float, occl: int, trunc: int, shift: float) -> str:
    tags = []
    if h < SMALL_PX:
        tags.append(f"small ({h:.0f} px)")
    if occl >= 1:
        tags.append("heavy occlusion" if occl == 2 else "partial occlusion")
    if trunc >= 1:
        tags.append("truncated at frame edge")
    if shift > h:
        # Beyond its own height, an uncompensated box no longer overlaps its next position.
        tags.append(f"camera shift {shift:.0f} px, more than the box height")
    return "; ".join(tags) or "no tag applies"


def sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)
    runs = REPO_ROOT / "cache" / "visdrone" / "runs" / args.run / "val"
    out = REPO_ROOT / "results" / "failures" / args.run
    out.mkdir(parents=True, exist_ok=True)
    shift_cache: dict[str, np.ndarray] = {}

    def shifts(seq: str) -> np.ndarray:
        if seq not in shift_cache:
            # Two frames at a time: some sequences are 4K, so a whole sequence does not fit.
            s, prev = [0.0], None
            for p in frame_paths(seq):
                img = cv2.imread(str(p))
                if prev is not None:
                    s.append(camera_shift(prev, img))
                prev = img
            shift_cache[seq] = np.array(s)
        return shift_cache[seq]

    summary = {}
    for g in ("person", "vehicle"):
        ev = pd.read_csv(runs / f"events_{g}.csv")
        sw = ev[ev["Type"] == "SWITCH"]
        worst = sw["sequence"].value_counts()
        picked = sw[sw["sequence"].isin(worst.index[:2])].head(args.top)
        rows_sw, tiles = [], []
        for _, e in picked.iterrows():
            seq, f, oid = e["sequence"], int(e["FrameId"]), int(e["OId"])
            fp = frame_paths(seq)
            gt = np.loadtxt(PREP / seq / f"gt_{g}.txt", delimiter=",", ndmin=2)
            pr = np.loadtxt(runs / f"{seq}_{g}.txt", delimiter=",", ndmin=2)
            box = gt[(gt[:, 0] == f) & (gt[:, 1] == oid)][0, 2:6]
            raw = raw_annotations(seq)
            near = raw[(raw["id"] == oid) & raw["frame"].between(f - 3, f)]
            occl, trunc = int(near["occl"].max()), int(near["trunc"].max())
            sh = shifts(seq)
            why = cause(box[3], occl, trunc, sh[f - 1])
            for ff, tag in ((f - 1, "before"), (f, "switch")):
                img = cv2.imread(str(fp[ff - 1]))
                tiles.append(
                    crop(img, box, boxes_at(gt, ff), boxes_at(pr, ff), f"{seq[3:11]} f{ff} {tag}")
                )
            rows_sw.append(
                {
                    "sequence": seq,
                    "frame": f,
                    "gt_id": oid,
                    "track_id": e["HId"],
                    "box_h_px": box[3],
                    "occlusion": occl,
                    "truncation": trunc,
                    "camera_shift_px": round(float(sh[f - 1]), 1),
                    "cause": why,
                }
            )
        if tiles:
            cv2.imwrite(
                str(out / f"switches_{g}.jpg"), sheet(tiles, 6), [cv2.IMWRITE_JPEG_QUALITY, 88]
            )
        pd.DataFrame(rows_sw).to_csv(out / f"switches_{g}.csv", index=False)

        # Most-missed small objects.
        cand = []
        for seq in sorted(ev["sequence"].unique()):
            gt = np.loadtxt(PREP / seq / f"gt_{g}.txt", delimiter=",", ndmin=2)
            if not len(gt):
                continue
            e = ev[(ev["sequence"] == seq) & ev["OId"].notna()]
            for oid, ge in e.groupby("OId"):
                rows = gt[gt[:, 1] == oid]
                if len(rows) < 20 or np.median(rows[:, 5]) >= SMALL_PX:
                    continue
                miss = (ge["Type"] == "MISS").mean()
                cand.append((miss, seq, int(oid), float(np.median(rows[:, 5])), len(rows)))
        cand.sort(reverse=True)
        rows_m, tiles = [], []
        for miss, seq, oid, hmed, n in cand[: args.top]:
            gt = np.loadtxt(PREP / seq / f"gt_{g}.txt", delimiter=",", ndmin=2)
            pr = np.loadtxt(runs / f"{seq}_{g}.txt", delimiter=",", ndmin=2)
            e = ev[(ev["sequence"] == seq) & (ev["OId"] == oid) & (ev["Type"] == "MISS")]
            f = int(e["FrameId"].iloc[len(e) // 2])
            box = gt[(gt[:, 0] == f) & (gt[:, 1] == oid)][0, 2:6]
            raw = raw_annotations(seq)
            r = raw[(raw["id"] == oid) & (raw["frame"] == f)]
            occl, trunc = int(r["occl"].max()), int(r["trunc"].max())
            sh = shifts(seq)
            why = cause(box[3], occl, trunc, sh[f - 1])
            img = cv2.imread(str(frame_paths(seq)[f - 1]))
            tiles.append(
                crop(
                    img, box, boxes_at(gt, f), boxes_at(pr, f), f"{seq[3:11]} f{f} miss {miss:.0%}"
                )
            )
            rows_m.append(
                {
                    "sequence": seq,
                    "gt_id": oid,
                    "frames": n,
                    "median_h_px": hmed,
                    "miss_share": round(miss, 3),
                    "shown_frame": f,
                    "occlusion": occl,
                    "truncation": trunc,
                    "camera_shift_px": round(float(sh[f - 1]), 1),
                    "cause": why,
                }
            )
        if tiles:
            cv2.imwrite(
                str(out / f"missed_small_{g}.jpg"), sheet(tiles, 6), [cv2.IMWRITE_JPEG_QUALITY, 88]
            )
        pd.DataFrame(rows_m).to_csv(out / f"missed_small_{g}.csv", index=False)
        summary[g] = {
            "switches_by_sequence": worst.to_dict(),
            "small_objects_considered": len(cand),
        }
        print(
            g, "switches by sequence:", worst.to_dict(), "| small objects:", len(cand), flush=True
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
