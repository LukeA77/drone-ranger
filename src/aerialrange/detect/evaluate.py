"""Detection average precision per evaluation group and object-size bin.

COCO-style: greedy matching in descending score order, each ground-truth box matched at most
once at IoU >= threshold; AP is the area under the interpolated precision-recall curve at 101
recall points. Size bins restrict ground truth by box height (pixels); detections matched to
out-of-bin ground truth, or unmatched detections whose height is outside the bin, are
ignored rather than counted as false positives (as COCO does for area ranges).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from aerialrange.track.evaluate import iou_distance

RECALL_POINTS = np.linspace(0, 1, 101)


def match_image(
    gt: NDArray, det: NDArray, scores: NDArray, iou_thr: float
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """True-positive flag per detection (in the given order) and matched gt index (-1)."""
    order = np.argsort(-scores, kind="mergesort")
    d = iou_distance(gt, det, iou_thr)  # 1 - IoU, NaN below threshold
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(det), bool)
    match = np.full(len(det), -1)
    for j in order:
        if len(gt) == 0:
            break
        col = np.where(used, np.nan, d[:, j])
        if np.all(np.isnan(col)):
            continue
        i = int(np.nanargmin(col))
        used[i] = True
        tp[j] = True
        match[j] = i
    return tp, match


def average_precision(recall: NDArray, precision: NDArray) -> float:
    """101-point interpolated AP from a PR curve ordered by decreasing score."""
    if len(recall) == 0:
        return 0.0
    prec_env = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    vals = np.where(idx < len(prec_env), prec_env[np.minimum(idx, len(prec_env) - 1)], 0.0)
    return float(vals.mean())


def evaluate_detections(
    gt: Mapping[str, NDArray],
    det: Mapping[str, tuple[NDArray, NDArray]],
    iou_thr: float,
    height_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """AP for one group. `gt[img]`: (n, 4) ltwh; `det[img]`: ((m, 4) ltwh, (m,) scores)."""
    all_scores, all_tp, n_gt = [], [], 0
    lo, hi = height_range if height_range else (-np.inf, np.inf)
    for img, g in gt.items():
        boxes, scores = det.get(img, (np.zeros((0, 4)), np.zeros(0)))
        in_bin = (g[:, 3] >= lo) & (g[:, 3] < hi)
        n_gt += int(in_bin.sum())
        tp, match = match_image(g, boxes, scores, iou_thr)
        keep = np.ones(len(boxes), bool)
        if height_range:
            # Images without ground truth have no matches (and an empty in_bin to index).
            matched_out = tp & ~in_bin[np.maximum(match, 0)] if len(g) else np.zeros_like(tp)
            unmatched_out = ~tp & ~((boxes[:, 3] >= lo) & (boxes[:, 3] < hi))
            keep = ~(matched_out | unmatched_out)
        all_scores.append(scores[keep])
        all_tp.append(tp[keep])
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tps = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    if n_gt == 0:
        return {
            "ap": float("nan"),
            "n_gt": 0,
            "n_det": int(len(scores)),
            "max_recall": float("nan"),
        }
    order = np.argsort(-scores, kind="mergesort")
    ctp = np.cumsum(tps[order])
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(order) + 1)
    return {
        "ap": average_precision(recall, precision),
        "n_gt": n_gt,
        "n_det": int(len(scores)),
        "max_recall": float(recall[-1]) if len(recall) else 0.0,
    }
