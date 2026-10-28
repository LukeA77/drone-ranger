"""Tiled inference for small objects (baseline B3).

The image is split into overlapping square tiles (`{{TILE_SIZE}}` px, overlap fraction
`{{TILE_OVERLAP}}`, chosen on validation); each tile is detected separately, boxes are mapped
back to full-image coordinates and merged with class-wise non-maximum suppression. The
latency cost is reported with the accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from aerialrange.track.evaluate import iou_distance

TileDetector = Callable[[NDArray, int, int], tuple[NDArray, NDArray, NDArray]]


def tile_grid(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Tiles (x0, y0, w, h) covering the image; the last tile in each row/column is aligned
    to the image edge so no tile extends past it."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        step = tile * (1 - overlap)
        k = math.ceil((n - tile) / step) + 1
        return sorted({min(round(i * step), n - tile) for i in range(k)})

    tw, th = min(tile, width), min(tile, height)
    return [(x, y, tw, th) for y in starts(height) for x in starts(width)]


def _ios(a: NDArray, B: NDArray) -> NDArray:
    """Intersection over the smaller box's area, between box `a` and each row of `B` (ltwh)."""
    iw = np.clip(np.minimum(a[0] + a[2], B[:, 0] + B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[1] + a[3], B[:, 1] + B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    smaller = np.minimum(a[2] * a[3], B[:, 2] * B[:, 3])
    return np.where(smaller > 0, iw * ih / np.where(smaller > 0, smaller, 1.0), 0.0)


def nms(
    boxes_ltwh: NDArray,
    scores: NDArray,
    cls: NDArray,
    iou_thr: float,
    ios_thr: float | None = None,
) -> NDArray:
    """Indices kept by greedy class-wise non-maximum suppression.

    With `ios_thr`, a lower-scoring box is also suppressed when at least that fraction of the
    smaller box lies inside the kept box (intersection over smaller area). This removes
    fragments of one object cut at tile edges, which plain IoU keeps (D-318)."""
    keep = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="mergesort")]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            rest = idx[1:]
            d = iou_distance(boxes_ltwh[i : i + 1], boxes_ltwh[rest], iou_thr)[0]
            survive = np.isnan(d)  # NaN distance means IoU below threshold
            if ios_thr is not None:
                survive &= _ios(boxes_ltwh[i], boxes_ltwh[rest]) < ios_thr
            idx = rest[survive]
    return np.array(sorted(keep), int)


def detect_tiled(
    img: NDArray,
    detector: TileDetector,
    tile: int,
    overlap: float,
    nms_iou: float,
    include_full: bool = False,
    nms_ios: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Run `detector(tile_image, x0, y0)` on every tile; returns merged (ltwh, scores, cls)
    in full-image coordinates. With `include_full`, the whole frame is also detected and merged
    in, so objects larger than a tile (split by tiling) are still found (D-317)."""
    h, w = img.shape[:2]
    B, S, C = [], [], []
    regions = tile_grid(w, h, tile, overlap)
    if include_full:
        regions = [(0, 0, w, h), *regions]
    for x0, y0, tw, th in regions:
        b, s, c = detector(img[y0 : y0 + th, x0 : x0 + tw], x0, y0)
        if len(b):
            b = np.asarray(b, float).copy()
            b[:, 0] += x0
            b[:, 1] += y0
            B.append(b)
            S.append(np.asarray(s, float))
            C.append(np.asarray(c, int))
    if not B:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b, s, c = np.concatenate(B), np.concatenate(S), np.concatenate(C)
    k = nms(b, s, c, nms_iou, ios_thr=nms_ios)
    return b[k], s[k], c[k]
