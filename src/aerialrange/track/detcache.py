"""Saved per-frame detections, replayed into a tracker (validation grid, `{{TRACKER_GRID}}`).

Detection is the slow stage, and tracker settings do not change it, so each detector
configuration is run once per sequence and its boxes saved. The tracker grid then replays
them with `run_sequence_custom`. Each frame keeps its image size, and replay checks it, so a
cache is never fed to frames of a different sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

Dets = tuple[np.ndarray, np.ndarray, np.ndarray]  # ltwh boxes, scores, class ids


def save_sequence(path: Path, dets: Sequence[Dets], shapes: Sequence[tuple[int, int]]) -> None:
    """Write one sequence's detections (one entry per frame, in frame order) to `path` (.npz)."""
    if len(dets) != len(shapes):
        raise ValueError("one image shape per frame")
    counts = np.array([len(s) for _, s, _ in dets], int)

    def cat(i: int, w: int) -> np.ndarray:
        return np.concatenate([np.asarray(d[i], float).reshape(-1, w) for d in dets])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=counts,
        ltwh=cat(0, 4),
        conf=cat(1, 1)[:, 0],
        cls=cat(2, 1)[:, 0].astype(int),
        shapes=np.asarray(shapes, int).reshape(-1, 2),
    )


def load_sequence(path: Path) -> tuple[list[Dets], np.ndarray]:
    """Per-frame detections and the (height, width) of each frame."""
    z = np.load(path)
    edges = np.concatenate([[0], np.cumsum(z["counts"])])
    dets = [
        (z["ltwh"][a:b], z["conf"][a:b], z["cls"][a:b])
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    return dets, z["shapes"]


class Replay:
    """A detection function for `run_sequence_custom` that returns saved frames in order."""

    def __init__(self, dets: Sequence[Dets], shapes: np.ndarray, min_conf: float = 0.0) -> None:
        self.dets, self.shapes, self.min_conf, self.k = dets, shapes, min_conf, 0

    def __call__(self, img: np.ndarray) -> Dets:
        if self.k >= len(self.dets):
            raise IndexError("more frames than cached detections")
        if tuple(img.shape[:2]) != tuple(self.shapes[self.k]):
            raise ValueError(f"frame {self.k + 1}: image size differs from the cache")
        b, s, c = self.dets[self.k]
        self.k += 1
        m = s >= self.min_conf
        return b[m], s[m], c[m]

    def skip(self) -> None:
        """Pass over the next frame without returning its detections (detection stride)."""
        if self.k >= len(self.dets):
            raise IndexError("more frames than cached detections")
        self.k += 1
