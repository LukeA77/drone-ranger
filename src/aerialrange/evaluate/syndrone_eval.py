"""SynDrone evaluation: truth tables, matching with ignore regions, pooled metrics.

Truth classes follow the version 1 VisDrone groups: person = walkers (tag 40); vehicle =
car, truck, bus (100, 101, 102). Riders, two-wheelers and trams/trains (41, 103-105) and
map ("level") boxes are ignored: a prediction that matches one is neither a true nor a false
positive. A person or vehicle actor that is too occluded or too small to count
(`TruthRules`) is ignored the same way, so the detector is neither required to find it nor
penalised for finding it, as are walkers stacked on one spot by a spawn glitch (D-712).
Ignored predictions are found as in TrackEval's MOT preprocessing: each frame, predictions
are matched to all truth boxes, valid and ignored, by the Hungarian algorithm at IoU >= 0.5,
and predictions matched to an ignored box are removed. The same removal feeds detection AP,
CLEAR/identity metrics and HOTA.

Every metric is returned as additive counts per sequence, so pooled values and bootstrap
intervals that resample whole sequences are exact recomputations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from aerialrange.track.evaluate import iou_distance

GROUPS = {"person": (40,), "vehicle": (100, 101, 102)}
IGNORE_TAGS = (41, 103, 104, 105)
SIZE_BINS_PX = (0, 16, 32, 64, 128, np.inf)  # box height, as version 1 (D-317)


@dataclass(frozen=True)
class TruthRules:
    min_visibility: float
    min_instance_px: int
    # Walkers closer than this (horizontal, footprint centres) to another walker in the same
    # frame are a simulator spawn stack, not people (D-712): ignored.
    walker_stack_m: float = 0.075


def walker_stacks(df: pd.DataFrame, radius_m: float) -> pd.Series:
    """True for walker rows whose footprint centre is within `radius_m` (horizontal) of
    another walker in the same frame. Uses every row of the frame (all cameras), so a stack
    is found even where one camera sees only one of its walkers."""
    from scipy.spatial import cKDTree

    out = pd.Series(False, index=df.index)
    w = df[(df["tag"] == 40) & ~df["level"].astype(bool)]
    if not len(w):
        return out
    stacked: set[tuple[int, int]] = set()
    pos = w.drop_duplicates(["frame", "id"])[["frame", "id", "foot_x", "foot_y"]]
    for f, g in pos.groupby("frame"):
        if len(g) < 2:
            continue
        pairs = cKDTree(g[["foot_x", "foot_y"]].to_numpy()).query_pairs(radius_m)
        ids = g["id"].to_numpy()
        for i, j in pairs:
            stacked.update({(int(f), int(ids[i])), (int(f), int(ids[j]))})
    if stacked:
        key = list(zip(w["frame"].astype(int), w["id"].astype(int), strict=True))
        out.loc[w.index] = [k in stacked for k in key]
    return out


def group_of(tag: int) -> str | None:
    for g, tags in GROUPS.items():
        if tag in tags:
            return g
    return None


def classify_truth(df: pd.DataFrame, rules: TruthRules) -> pd.DataFrame:
    """Add `group` (person / vehicle / None) and `status` (valid / ignore / absent).

    absent: no instance pixels at all (out of view or fully hidden): not truth, not ignore.
    """
    out = df.copy()
    out["group"] = pd.Series([group_of(int(t)) for t in out["tag"]], index=out.index, dtype=object)
    seen = out["n_instance_px"] > 0
    counts = (
        out["group"].notna()
        & ~out["level"].astype(bool)
        & (out["visibility"] >= rules.min_visibility)
        & (out["n_instance_px"] >= rules.min_instance_px)
    )
    if {"foot_x", "foot_y"} <= set(out.columns):
        counts &= ~walker_stacks(out, rules.walker_stack_m)
    out["status"] = np.where(counts, "valid", np.where(seen, "ignore", "absent"))
    return out


def ltwh(b: NDArray) -> NDArray:
    """[l, t, r, b] boxes to [l, t, w, h]."""
    b = np.asarray(b, float).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


def iou_matrix(a_ltwh: NDArray, b_ltwh: NDArray) -> NDArray:
    d = iou_distance(a_ltwh, b_ltwh, 0.0)
    return 1.0 - np.nan_to_num(d, nan=1.0) if d.size else np.zeros((len(a_ltwh), len(b_ltwh)))


def keep_not_ignored(
    pred: NDArray, valid: NDArray, ignored: NDArray, iou_thr: float = 0.5
) -> NDArray[np.bool_]:
    """Mask of predictions (n, 4 ltwh) to keep: drop those the Hungarian match over valid and
    ignored truth assigns to an ignored box at IoU >= iou_thr.

    Ties go to valid truth: a map box can sit exactly on a valid actor (4 to 8 % of valid
    views on the development towns), and an arbitrary tie-break would then discard the
    prediction and count the actor as missed (D-711)."""
    n = len(pred)
    if n == 0 or len(ignored) == 0:
        return np.ones(n, bool)
    gt = np.vstack([valid.reshape(-1, 4), ignored.reshape(-1, 4)])
    iou = iou_matrix(gt, pred)
    nv = len(valid)
    bonus = np.zeros_like(iou)
    bonus[:nv] = 1e-6  # far below any meaningful IoU difference
    rows, cols = linear_sum_assignment(-(iou + bonus))
    keep = np.ones(n, bool)
    for r, c in zip(rows, cols, strict=True):
        if iou[r, c] >= iou_thr and r >= nv:
            keep[c] = False
    return keep


def match_pairs(gt: NDArray, pred: NDArray, iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Hungarian one-to-one matches (gt index, pred index, IoU) with IoU >= iou_thr."""
    if len(gt) == 0 or len(pred) == 0:
        return []
    iou = iou_matrix(gt, pred)
    rows, cols = linear_sum_assignment(-iou)
    return [
        (int(r), int(c), float(iou[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if iou[r, c] >= iou_thr
    ]


def interval_coverage(
    err: NDArray, sigma: NDArray, levels: tuple[float, ...] = (0.68, 0.90, 0.95)
) -> dict[str, float]:
    """Share of |err| <= z * sigma for two-sided Gaussian intervals at each nominal level."""
    from scipy.stats import norm

    err, sigma = np.asarray(err, float), np.asarray(sigma, float)
    ok = np.isfinite(err) & np.isfinite(sigma)
    out = {}
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2)
        out[f"{lv:.2f}"] = float(np.mean(np.abs(err[ok]) <= z * sigma[ok])) if ok.any() else np.nan
    return out


def bootstrap(
    units: list, stat: Callable[[list], float | None], n: int, seed: int
) -> tuple[float, float]:
    """95 % percentile interval of `stat(resampled units)`, resampling whole units."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(units), len(units))
        v = stat([units[i] for i in idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
