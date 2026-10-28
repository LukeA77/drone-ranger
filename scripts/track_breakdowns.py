"""Breakdowns of a tracking run: object size, crowd density, camera motion.

Usage: python scripts/track_breakdowns.py --run B2_yolo26n_960_tuned_botsort_cmc --split test-dev
       [--final]

Reads the predictions a run already wrote (cache/visdrone/runs/<run>/<split>/) and the prepared
ground truth; nothing is re-detected or re-tracked, so analysing the one-time test-dev run does
not run it again; `--final` is still required for test-dev. Per group:
- size: ground-truth objects by box height (the bins used throughout), with the share matched
  (recall) and the identity-switch rate per matched object-frame;
- crowd density: per sequence, mean ground-truth objects per frame against IDF1 and MOTA;
- camera motion: per sequence, median frame-to-frame image shift (phase correlation on
  quarter-size frames, full-size pixels) against IDF1 and MOTA.
Spearman rank correlations across sequences are given with the number of sequences (17 on
test-dev, so they are indicative only). Output: results/visdrone/<split>/breakdowns_<run>.json
and figures in results/visdrone/<split>/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd

from aerialrange.track.evaluate import accumulate, read_ignored, read_rows, summarise
from aerialrange.utils.config import REPO_ROOT, load_config
from aerialrange.viz.plots import line_chart

HEIGHT_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]


def camera_shift(frames: list) -> float:
    s, prev, out = 0.25, None, []
    for p in frames:
        g = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), None, fx=s, fy=s).astype(
            np.float32
        )
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            out.append(np.hypot(dx, dy) / s)
        prev = g
    return float(np.median(out)) if out else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    if args.split == "test-dev" and not args.final:
        print("test-dev analysis needs --final (it reads the one-time run's outputs).")
        return 1
    iou = load_config("detector").require("iou_match")
    prep = REPO_ROOT / "cache" / "visdrone" / "mot" / args.split
    runs = REPO_ROOT / "cache" / "visdrone" / "runs" / args.run / args.split
    out_dir = REPO_ROOT / "results" / "visdrone" / args.split
    seqs = sorted(p for p in prep.iterdir() if p.is_dir())
    res: dict = {"run": args.run, "split": args.split, "groups": {}}
    seq_rows = []
    shifts = {}
    for seq in seqs:
        info = json.loads((seq / "seqinfo.json").read_text())
        frames = sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
        shifts[seq.name] = camera_shift(frames)
        print(f"{seq.name}: camera shift {shifts[seq.name]:.1f} px", flush=True)
    for g in ("person", "vehicle"):
        size_tot = np.zeros(len(HEIGHT_BINS))
        size_match = np.zeros(len(HEIGHT_BINS))
        size_sw = np.zeros(len(HEIGHT_BINS))
        per_seq = {}
        for seq in seqs:
            gt = read_rows(seq / f"gt_{g}.txt", 6)
            pr = (
                read_rows(runs / f"{seq.name}_{g}.txt", 7)
                if (runs / f"{seq.name}_{g}.txt").exists()
                else np.zeros((0, 7))
            )
            acc = accumulate(gt, pr, read_ignored(seq / "ignored.txt"), iou)
            per_seq[seq.name] = acc
            ev = acc.mot_events.reset_index()
            ev = ev[ev["Type"].isin(["MATCH", "SWITCH", "MISS"])]
            if not len(gt) or not len(ev):
                continue
            hmap = {
                (int(f), int(o)): h for f, o, h in zip(gt[:, 0], gt[:, 1], gt[:, 5], strict=True)
            }
            hs = np.array(
                [
                    hmap.get((int(f), int(o)), np.nan)
                    for f, o in zip(ev["FrameId"], ev["OId"], strict=True)
                ]
            )
            for b, (lo, hi) in enumerate(HEIGHT_BINS):
                m = (hs >= lo) & (hs < hi)
                size_tot[b] += m.sum()
                size_match[b] += (m & ev["Type"].isin(["MATCH", "SWITCH"]).to_numpy()).sum()
                size_sw[b] += (m & (ev["Type"] == "SWITCH").to_numpy()).sum()
        metrics = summarise(per_seq)
        sizes = {}
        for b, (lo, hi) in enumerate(HEIGHT_BINS):
            key = f"{lo}-{hi if hi < 1e9 else 'inf'}"
            sizes[key] = {
                "gt_object_frames": int(size_tot[b]),
                "recall": float(size_match[b] / size_tot[b]) if size_tot[b] else None,
                "switches_per_1000_matched": float(1000 * size_sw[b] / size_match[b])
                if size_match[b]
                else None,
            }
        rows = []
        for seq in seqs:
            gt = read_rows(seq / f"gt_{g}.txt", 6)
            n_frames = json.loads((seq / "seqinfo.json").read_text())["frames"]
            m = metrics.get(seq.name)
            if m is None or not len(gt):
                continue
            rows.append(
                {
                    "sequence": seq.name,
                    "group": g,
                    "objects_per_frame": len(gt) / n_frames,
                    "camera_shift_px": shifts[seq.name],
                    "idf1": m["idf1"],
                    "mota": m["mota"],
                }
            )
        df = pd.DataFrame(rows)
        seq_rows += rows
        res["groups"][g] = {
            "overall": metrics["OVERALL"],
            "by_box_height": sizes,
            "density_vs_idf1_spearman": spearman(df["objects_per_frame"], df["idf1"])
            if len(df) > 2
            else None,
            "density_vs_mota_spearman": spearman(df["objects_per_frame"], df["mota"])
            if len(df) > 2
            else None,
            "camera_shift_vs_idf1_spearman": spearman(df["camera_shift_px"], df["idf1"])
            if len(df) > 2
            else None,
            "camera_shift_vs_mota_spearman": spearman(df["camera_shift_px"], df["mota"])
            if len(df) > 2
            else None,
            "n_sequences": int(len(df)),
        }
        print(g, {k: v["recall"] for k, v in sizes.items()}, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"breakdowns_{args.run}.json").write_text(
        json.dumps(res, indent=1, default=float) + "\n"
    )
    pd.DataFrame(seq_rows).to_csv(out_dir / f"breakdowns_{args.run}_sequences.csv", index=False)
    mids = [8, 24, 48, 96, 192]
    line_chart(
        out_dir / "figures" / f"recall_by_height_{args.run}.png",
        {
            g: {
                "x": mids,
                "y": [
                    res["groups"][g]["by_box_height"][k]["recall"] or float("nan")
                    for k in res["groups"][g]["by_box_height"]
                ],
            }
            for g in ("person", "vehicle")
        },
        f"Share of objects tracked, by box height ({args.split})",
        "Box height (px, bin centre: 0-16, 16-32, 32-64, 64-128, 128+)",
        "Recall (matched object-frames)",
        logx=True,
        note="Tracker output against ground truth at IoU 0.5; ignored regions excluded.",
    )
    print(
        json.dumps(
            {
                g: {k: v for k, v in res["groups"][g].items() if k != "overall"}
                for g in res["groups"]
            },
            indent=1,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
