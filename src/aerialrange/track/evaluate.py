"""Tracking evaluation with py-motmetrics.

Per evaluation group (person, vehicle): ground truth from `scripts/visdrone_prep.py --mot`,
predictions as (frame, id, left, top, width, height[, score]) rows. Predictions whose centre
lies in an ignored region of that frame are removed before scoring. Matching
uses IoU >= `{{IOU_MATCH}}`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import motmetrics as mm
import numpy as np
from numpy.typing import NDArray

from aerialrange.data.visdrone import outside_ignored

METRICS = [
    "mota",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_switches",
    "num_fragmentations",
    "mostly_tracked",
    "mostly_lost",
    "num_objects",
    "num_false_positives",
    "num_misses",
]


def read_rows(path: Path, n_cols: int) -> NDArray:
    """Comma-separated numeric rows (empty file gives shape (0, n_cols))."""
    text = path.read_text().strip()
    if not text:
        return np.zeros((0, n_cols))
    return np.array([[float(x) for x in ln.split(",")[:n_cols]] for ln in text.splitlines()])


def read_ignored(path: Path) -> dict[int, NDArray]:
    """Ignored regions per frame: {frame: (k, 4) ltwh}."""
    rows = read_rows(path, 5)
    out: dict[int, list] = {}
    for f, *box in rows:
        out.setdefault(int(f), []).append(box)
    return {k: np.array(v) for k, v in out.items()}


def iou_distance(a_ltwh: NDArray, b_ltwh: NDArray, iou_match: float) -> NDArray:
    """1 - IoU between box sets, NaN where IoU < `iou_match` (no match allowed).

    Our own implementation: py-motmetrics 1.4.0 `distances.iou_matrix` calls `np.asfarray`,
    which NumPy 2 removed (D-306).
    """
    a = np.asarray(a_ltwh, float).reshape(-1, 4)
    b = np.asarray(b_ltwh, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    return np.where(iou >= iou_match, 1.0 - iou, np.nan)


def accumulate(
    gt: NDArray, pred: NDArray, ignored: Mapping[int, NDArray], iou_match: float
) -> mm.MOTAccumulator:
    """MOT accumulator for one sequence and group. `gt`, `pred`: rows of
    (frame, id, left, top, width, height, ...)."""
    acc = mm.MOTAccumulator(auto_id=False)
    frames = np.union1d(gt[:, 0], pred[:, 0]).astype(int) if len(gt) + len(pred) else []
    for f in frames:
        g = gt[gt[:, 0] == f]
        p = pred[pred[:, 0] == f]
        if f in ignored and len(p):
            p = p[outside_ignored(p[:, 2:6], ignored[f])]
        dist = iou_distance(g[:, 2:6], p[:, 2:6], iou_match)
        acc.update(g[:, 1].astype(int).tolist(), p[:, 1].astype(int).tolist(), dist, frameid=f)
    return acc


def summarise(accs: Mapping[str, mm.MOTAccumulator]) -> dict[str, dict[str, Any]]:
    """Metrics per sequence plus an overall row ("OVERALL")."""
    mh = mm.metrics.create()
    names = list(accs)
    df = mh.compute_many(
        [accs[n] for n in names], metrics=METRICS, names=names, generate_overall=True
    )
    return {str(idx): {k: float(v) for k, v in row.items()} for idx, row in df.iterrows()}


HEIGHT_BINS = ((0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9))  # px, used throughout


def height_key(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}-inf"


def counts_by_height(
    acc: mm.MOTAccumulator, gt: NDArray, bins: Sequence[tuple[float, float]] = HEIGHT_BINS
) -> NDArray:
    """Ground-truth object-frames per height bin: rows (total, matched, switches), one column
    per bin. Matched counts matches and identity switches; objects in ignored regions are
    already out of `acc`. `gt` rows are (frame, id, left, top, width, height)."""
    out = np.zeros((3, len(bins)))
    ev = acc.mot_events.reset_index()
    ev = ev[ev["Type"].isin(["MATCH", "SWITCH", "MISS"])]
    if not len(gt) or not len(ev):
        return out
    hmap = {(int(f), int(o)): h for f, o, h in zip(gt[:, 0], gt[:, 1], gt[:, 5], strict=True)}
    hs = np.array(
        [hmap.get((int(f), int(o)), np.nan) for f, o in zip(ev["FrameId"], ev["OId"], strict=True)]
    )
    typ = ev["Type"].to_numpy()
    for b, (lo, hi) in enumerate(bins):
        m = (hs >= lo) & (hs < hi)
        out[0, b] = m.sum()
        out[1, b] = (m & np.isin(typ, ["MATCH", "SWITCH"])).sum()
        out[2, b] = (m & (typ == "SWITCH")).sum()
    return out
