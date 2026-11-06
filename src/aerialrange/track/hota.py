"""Higher Order Tracking Accuracy (HOTA; Luiten et al., 2021, IJCV 129, 548-578).

Vendored from TrackEval (`trackeval/metrics/hota.py`, commit 12c8791, 2022-11-29), MIT
licence, Copyright (c) 2020 Jonathon Luiten; notice reproduced below as the licence requires.
Changes: the removed NumPy aliases `np.float` / `np.int` are replaced by `float` (TrackEval
fails on NumPy 2, which this project uses); the class wrapper, timing and plotting are
dropped; inputs are plain arrays. The algorithm and every formula are unchanged, checked by
`tests/test_hota.py`.

    MIT License

    Copyright (c) 2020 Jonathon Luiten

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify,
    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be included in all copies
    or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)
_EPS = np.finfo("float").eps
INT_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")


def _final(res: dict) -> dict:
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = res["HOTA"][0]
    res["LocA(0)"] = res["LocA"][0]
    res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
    return res


def hota_sequence(
    gt_ids: Sequence[NDArray], tracker_ids: Sequence[NDArray], similarity: Sequence[NDArray]
) -> dict:
    """HOTA for one sequence.

    Per timestep t: `gt_ids[t]` and `tracker_ids[t]` are integer ids remapped to 0..n-1
    over the sequence; `similarity[t]` is the (len(gt_ids[t]), len(tracker_ids[t])) IoU
    matrix. Ignored objects and predictions must already be removed.
    """
    na = len(ALPHAS)
    res: dict = {
        f: np.zeros(na, dtype=float)
        for f in (
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
            *INT_FIELDS,
        )
    }
    n_gt_dets = int(sum(len(g) for g in gt_ids))
    n_tr_dets = int(sum(len(t) for t in tracker_ids))
    if n_tr_dets == 0:
        res["HOTA_FN"] = n_gt_dets * np.ones(na, dtype=float)
        res["LocA"] = np.ones(na, dtype=float)
        return _final(res)
    if n_gt_dets == 0:
        res["HOTA_FP"] = n_tr_dets * np.ones(na, dtype=float)
        res["LocA"] = np.ones(na, dtype=float)
        return _final(res)
    n_gt = int(max(int(g.max()) for g in gt_ids if len(g)) + 1)
    n_tr = int(max(int(t.max()) for t in tracker_ids if len(t)) + 1)
    potential = np.zeros((n_gt, n_tr))
    gt_count = np.zeros((n_gt, 1))
    tr_count = np.zeros((1, n_tr))
    for g, t, sim in zip(gt_ids, tracker_ids, similarity, strict=True):
        denom = sim.sum(0)[np.newaxis, :] + sim.sum(1)[:, np.newaxis] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 0 + _EPS
        sim_iou[mask] = sim[mask] / denom[mask]
        potential[g[:, np.newaxis], t[np.newaxis, :]] += sim_iou
        gt_count[g] += 1
        tr_count[0, t] += 1
    global_align = potential / (gt_count + tr_count - potential)
    matches = [np.zeros_like(potential) for _ in ALPHAS]
    for g, t, sim in zip(gt_ids, tracker_ids, similarity, strict=True):
        if len(g) == 0:
            res["HOTA_FP"] += len(t)
            continue
        if len(t) == 0:
            res["HOTA_FN"] += len(g)
            continue
        score = global_align[g[:, np.newaxis], t[np.newaxis, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - _EPS
            r, c = rows[ok], cols[ok]
            res["HOTA_TP"][a] += len(r)
            res["HOTA_FN"][a] += len(g) - len(r)
            res["HOTA_FP"][a] += len(t) - len(r)
            if len(r):
                res["LocA"][a] += sim[r, c].sum()
                matches[a][g[r], t[c]] += 1
    for a in range(na):
        m = matches[a]
        tp = np.maximum(1, res["HOTA_TP"][a])
        res["AssA"][a] = np.sum(m * (m / np.maximum(1, gt_count + tr_count - m))) / tp
        res["AssRe"][a] = np.sum(m * (m / np.maximum(1, gt_count))) / tp
        res["AssPr"][a] = np.sum(m * (m / np.maximum(1, tr_count))) / tp
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _final(res)


def hota_combine(per_seq: Mapping[str, dict]) -> dict:
    """Combine sequences as TrackEval does (sums, then HOTA_TP-weighted averages)."""
    res: dict = {f: sum(r[f] for r in per_seq.values()) for f in INT_FIELDS}
    for f in ("AssRe", "AssPr", "AssA"):
        res[f] = sum(r[f] * r["HOTA_TP"] for r in per_seq.values()) / np.maximum(
            1.0, res["HOTA_TP"]
        )
    loc = sum(r["LocA"] * r["HOTA_TP"] for r in per_seq.values())
    res["LocA"] = np.maximum(1e-10, loc) / np.maximum(1e-10, res["HOTA_TP"])
    return _final(res)


def summary(res: dict) -> dict[str, float]:
    """Headline numbers: HOTA, DetA, AssA, LocA averaged over alpha (as TrackEval reports)."""
    return {
        k: float(np.mean(res[k]))
        for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
    }
