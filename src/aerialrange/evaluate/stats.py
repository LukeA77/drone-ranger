"""Statistics for the evaluation.

Mid-Air frames within a trajectory are strongly correlated, so every interval comes from a
bootstrap that resamples whole groups (trajectories), not samples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, rankdata

Stat = Callable[[NDArray], float]


def group_bootstrap_ci(
    x: ArrayLike,
    groups: ArrayLike,
    stat: Stat,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of `stat(x)` from resampling whole groups with replacement."""
    x, groups = np.asarray(x), np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == i)[0] for i in range(len(ug))]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        vals[b] = stat(x[np.concatenate([members[i] for i in pick])])
    a = (1 - level) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def binned_stat(
    x: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    edges: Sequence[float],
    stat: Stat,
    n_boot: int,
    min_samples: int,
    rng: np.random.Generator,
    level: float = 0.95,
    min_groups: int = 1,
) -> list[dict[str, Any]]:
    """`stat(y)` per bin of `x` with a group-bootstrap interval. Bins with fewer than
    `min_samples` samples (`{{MIN_SAMPLES_PER_BIN}}`) or from fewer than `min_groups` groups
    (`{{MIN_GROUPS_PER_BIN}}`) are kept in the table but not reported."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        n = int(m.sum())
        row: dict[str, Any] = {
            "lo": float(lo),
            "hi": float(hi),
            "n": n,
            "n_groups": int(len(np.unique(groups[m]))),
        }
        if n >= min_samples and n > 0 and row["n_groups"] >= min_groups:
            row["stat"] = float(stat(y[m]))
            row["ci_lo"], row["ci_hi"] = group_bootstrap_ci(
                y[m], groups[m], stat, n_boot, rng, level
            )
            row["reported"] = True
        else:
            row.update(stat=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), reported=False)
        rows.append(row)
    return rows


def bootstrap_stability(
    x: ArrayLike, groups: ArrayLike, stat: Stat, n_boot: int, rng_seed: int
) -> dict[str, Any]:
    """Compare the interval at `n_boot` and `2 n_boot` resamples (the `{{N_BOOT}}` check)."""
    a = group_bootstrap_ci(x, groups, stat, n_boot, np.random.default_rng(rng_seed))
    b = group_bootstrap_ci(x, groups, stat, 2 * n_boot, np.random.default_rng(rng_seed + 1))
    width = max(b[1] - b[0], np.finfo(float).tiny)
    return {
        "n_boot": n_boot,
        "ci": a,
        "ci_double": b,
        "max_endpoint_change": float(max(abs(a[0] - b[0]), abs(a[1] - b[1])) / width),
    }


def z_halfwidth(level: float) -> float:
    """Half-width, in sigmas, of the central Gaussian interval with probability `level`."""
    return float(norm.ppf((1 + level) / 2))


def coverage(z: ArrayLike, levels: Sequence[float]) -> dict[float, float]:
    """Empirical coverage of nominal Gaussian intervals given standardised errors z."""
    z = np.abs(np.asarray(z, float))
    return {p: float(np.mean(z <= z_halfwidth(p))) for p in levels}


def fit_sigma_scale(errors: ArrayLike, sigmas: ArrayLike) -> float:
    """Maximum-likelihood scalar s for e ~ N(0, (s sigma)^2): s = sqrt(mean((e / sigma)^2))."""
    z = np.asarray(errors, float) / np.asarray(sigmas, float)
    return float(np.sqrt(np.mean(z**2)))


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney statistic, ties count one half)."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def precision_recall(scores: ArrayLike, labels: ArrayLike) -> dict[str, NDArray]:
    """Precision and recall when flagging every sample with score >= threshold, for each
    distinct score as threshold (decreasing)."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(s) + 1)
    last = np.r_[s[1:] != s[:-1], True]  # last index of each distinct threshold
    return {
        "threshold": s[last],
        "precision": (tp / k)[last],
        "recall": (tp / max(y.sum(), 1))[last],
    }


def threshold_for_recall(
    scores: ArrayLike, labels: ArrayLike, target_recall: float
) -> dict[str, float]:
    """Highest threshold whose recall reaches `target_recall`, with its precision."""
    if not np.asarray(labels).astype(bool).any():
        raise ValueError("no positive samples")
    pr = precision_recall(scores, labels)
    i = int(np.argmax(pr["recall"] >= target_recall))
    return {
        "threshold": float(pr["threshold"][i]),
        "precision": float(pr["precision"][i]),
        "recall": float(pr["recall"][i]),
    }


def threshold_max_f1(scores: ArrayLike, labels: ArrayLike) -> dict[str, float]:
    """Threshold with the highest F1 (flag when score >= threshold); no free parameter."""
    pr = precision_recall(scores, labels)
    p, r = pr["precision"], pr["recall"]
    f1 = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0.0)
    i = int(np.argmax(f1))
    return {
        "threshold": float(pr["threshold"][i]),
        "precision": float(p[i]),
        "recall": float(r[i]),
        "f1": float(f1[i]),
    }


def usable_limit(rows: Sequence[dict[str, Any]], threshold: float, direction: str) -> float | None:
    """Edge of the usable region from binned rows (with `lo`, `hi`, `reported`, `ci_hi`).

    direction="above" (error falls as x rises, e.g. depression): the smallest bin edge `lo`
    such that every reported bin from there upwards has its upper 95 % bound below
    `threshold`. direction="below" (error grows with x, e.g. range): the `lo` edge of the
    first reported bin that crosses the threshold (usable below it). None if nowhere usable.
    """
    rep = [r for r in rows if r["reported"]]
    if direction == "above":
        limit = None
        for r in reversed(rep):
            if r["ci_hi"] < threshold:
                limit = r["lo"]
            else:
                break
        return limit
    if direction == "below":
        if not rep or rep[0]["ci_hi"] >= threshold:
            return None
        for r in rep:
            if r["ci_hi"] >= threshold:
                return r["lo"]
        return rep[-1]["hi"]
    raise ValueError("direction must be 'above' or 'below'")
