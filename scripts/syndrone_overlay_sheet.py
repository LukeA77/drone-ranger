"""Contact sheet of pipeline output against SynDrone truth, for looking at, not scoring.

Usage: python scripts/syndrone_overlay_sheet.py --town Town10HD [--frames 1000 2700]

For each camera height (20, 50, 80 m) and each frame: the pipeline's tracked boxes
(`cache/syndrone/runs/<name>/<town>_h<h>/tracks.csv`), coloured by match, each labelled with
its range and the true range of the matched object (IoU >= 0.5 against valid truth, as in
`scripts/syndrone_evaluate.py`), plus speed (shown speed, or "settling") against true speed.
Valid truth objects with no matching track are drawn in red ("missed"). Each tile is the
busiest half-size window at full resolution, with the whole frame as an inset.

Writes `cache/syndrone/figures/overlay_<town>.jpg` (dataset imagery, not committed). Uses
the same matching as the evaluation; numbers here are illustrations, the scores are in
results/syndrone/development/.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerialrange.data.syndrone import SynDroneSequence
from aerialrange.data.syndrone_archive import MemberStore
from aerialrange.evaluate.syndrone_eval import TruthRules, classify_truth, iou_matrix
from aerialrange.utils.config import REPO_ROOT, load_config

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
TRUTH = REPO_ROOT / "cache" / "syndrone" / "truth"
SPEED = REPO_ROOT / "cache" / "syndrone" / "speed"
FIG = REPO_ROOT / "cache" / "syndrone" / "figures"
GREEN, AMBER, RED, WHITE, INK = (
    (60, 200, 60),
    (0, 190, 255),
    (40, 40, 230),
    (255, 255, 255),
    (0, 0, 0),
)


def _text(
    img: NDArray, s: str, org: tuple[int, int], scale: float = 0.5, col: tuple = WHITE
) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, INK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def tile(img: NDArray, trk: pd.DataFrame, tru: pd.DataFrame, speed: dict, title: str) -> NDArray:
    """Annotate one frame; return the busiest half-size window at full resolution."""
    valid = tru[tru["status"] == "valid"]
    tb = valid[["mask_left", "mask_top", "mask_right", "mask_bottom"]].to_numpy(float)
    p_ltwh = trk[["left", "top", "width", "height"]].to_numpy(float).reshape(-1, 4)
    t_ltwh = np.c_[tb[:, :2], tb[:, 2:] - tb[:, :2]] if len(tb) else np.zeros((0, 4))
    pb = np.c_[p_ltwh[:, :2], p_ltwh[:, :2] + p_ltwh[:, 2:]]  # corners, for drawing
    iou = (
        iou_matrix(p_ltwh, t_ltwh)
        if len(p_ltwh) and len(t_ltwh)
        else np.zeros((len(p_ltwh), len(t_ltwh)))
    )
    matched_t = set()
    out = img.copy()
    labels = []
    for i, (_, r) in enumerate(trk.iterrows()):
        x0, y0, x1, y1 = pb[i].astype(int)
        j = int(iou[i].argmax()) if iou.shape[1] else -1
        if j >= 0 and iou[i, j] >= 0.5 and j not in matched_t:
            matched_t.add(j)
            t = valid.iloc[j]
            col = GREEN
            s = f"{r['range_m']:.1f} m (true {t['range_m']:.1f})"
            v = speed.get((int(t["id"]), int(t["frame"])))
            if r["speed_shown"]:
                s += f" | {r['speed_ms']:.1f} m/s"
            elif np.isfinite(r["speed_ms"]):
                s += f" | settling ({r['speed_ms']:.1f} m/s)"
            if v is not None and np.isfinite(v):
                s += f" (true {v:.1f})"
        else:
            col = AMBER
            s = f"{r['range_m']:.1f} m (no valid truth match)"
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        labels.append((s, (x0, max(14, y0 - 4)), col))
    for j in set(range(len(tb))) - matched_t:
        x0, y0, x1, y1 = tb[j].astype(int)
        cv2.rectangle(out, (x0, y0), (x1, y1), RED, 2)
    for s, org, col in labels:
        _text(out, s, org, 0.5, col)
    # zoom on the 2x-scaled window with most boxes
    h, w = out.shape[:2]
    zw, zh = w // 2, h // 2
    centres = np.r_[pb[:, :2] + (pb[:, 2:] - pb[:, :2]) / 2, (tb[:, :2] + tb[:, 2:]) / 2]
    best, bxy = -1, (0, 0)
    for x in range(0, w - zw + 1, w // 8):
        for y in range(0, h - zh + 1, h // 8):
            n = (
                int(
                    (
                        (centres[:, 0] >= x)
                        & (centres[:, 0] < x + zw)
                        & (centres[:, 1] >= y)
                        & (centres[:, 1] < y + zh)
                    ).sum()
                )
                if len(centres)
                else 0
            )
            if n > best:
                best, bxy = n, (x, y)
    x, y = bxy
    # tile = the busiest half-size window at full resolution, so labels stay readable,
    # with the whole frame as an inset (white box = the window)
    t = out[y : y + zh, x : x + zw].copy()
    full = out.copy()
    cv2.rectangle(full, (x, y), (x + zw, y + zh), WHITE, 8)
    iw = zw // 4
    inset = cv2.resize(full, (iw, iw * h // w), interpolation=cv2.INTER_AREA)
    t[-inset.shape[0] - 6 : -6, -iw - 6 : -6] = inset
    cv2.rectangle(t, (zw - iw - 7, zh - inset.shape[0] - 7), (zw - 6, zh - 6), WHITE, 1)
    cv2.rectangle(t, (0, 0), (zw, 34), INK, -1)
    n_m, n_t = len(matched_t), len(tb)
    _text(t, f"{title}  |  valid truth tracked {n_m} of {n_t} (whole frame)", (8, 24), 0.65)
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--name", default="v1_accurate")
    ap.add_argument("--frames", type=int, nargs="+", default=[1000, 2700])
    args = ap.parse_args(argv)
    cfg = load_config("syndrone")
    rules = TruthRules(
        float(cfg.require("truth_min_visibility")), int(cfg.require("truth_min_instance_px"))
    )
    d = load_config("data")
    store = MemberStore(
        REPO_ROOT / d.require("data_root") / d.require("syndrone_subdir"),
        d.require("syndrone_base_url"),
    )
    tr = classify_truth(pd.read_csv(TRUTH / args.town / "objects.csv.gz"), rules)
    sp = pd.read_csv(SPEED / f"{args.town}.csv.gz")
    speed = dict(zip(zip(sp["id"], sp["frame"], strict=True), sp["speed_ms"], strict=True))
    rows = []
    for h in (20, 50, 80):
        trk = pd.read_csv(RUNS / args.name / f"{args.town}_h{h}" / "tracks.csv")
        seq = SynDroneSequence(store, args.town, h)
        tiles = []
        for k in args.frames:
            img = seq.rgb(k)
            t = trk[trk["frame"] - 1 == k]
            g = tr[(tr["height"] == h) & (tr["frame"] == k)]
            tiles.append(tile(img, t, g, speed, f"{args.town} {h} m, frame {k}"))
        rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    key = np.zeros((40, sheet.shape[1], 3), np.uint8)
    _text(
        key,
        "green: tracked, matched to truth (pipeline range, true range | speed)",
        (10, 26),
        0.6,
        GREEN,
    )
    _text(key, "amber: track with no valid truth match", (760, 26), 0.6, AMBER)
    _text(key, "red: valid truth object missed", (1180, 26), 0.6, RED)
    _text(key, "SynDrone (CC-BY, Rizzoli et al. 2023)", (sheet.shape[1] - 420, 26), 0.6)
    sheet = np.vstack([key, sheet])
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"overlay_{args.town}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
