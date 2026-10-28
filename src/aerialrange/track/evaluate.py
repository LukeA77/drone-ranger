"""Tracking evaluation with py-motmetrics.

Per evaluation group (person, vehicle): ground truth from `scripts/visdrone_prep.py --mot`,
predictions as (frame, id, left, top, width, height[, score]) rows. Predictions whose centre
lies in an ignored region of that frame are removed before scoring. Matching
uses IoU >= `{{IOU_MATCH}}`.
"""

from __future__ import annotations

from collections.abc import Mapping
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
