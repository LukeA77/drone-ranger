"""Split conformal calibration of range intervals (plan B).

Score each calibration sample by how many predicted sigmas its error was: s_i = |e_i| / sigma_i.
For coverage level p, the interval for a new estimate is +/- q_p * sigma, where q_p is the
ceil((n + 1) p)-th smallest calibration score (the finite-sample split conformal rule; Vovk,
Gammerman and Shafer, 2005; Lei et al., 2018). If calibration and new samples are exchangeable,
coverage is at least p. "Mondrian" conformal fits one q_p per group (for example a viewing
angle band), which gives coverage within each group; groups with fewer than `min_group`
samples use the global q_p.

Caveat recorded in docs/decisions.md (D-403): Mid-Air samples are correlated within a
trajectory, so the guarantee is weaker than for independent samples; coverage is checked
empirically on the held-out test split.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def conformal_quantile(scores: ArrayLike, level: float) -> float:
    """The ceil((n + 1) level)-th smallest score; +inf if that rank exceeds n."""
    s = np.sort(np.asarray(scores, float))
    k = math.ceil((len(s) + 1) * level)
    return float("inf") if k > len(s) else float(s[k - 1])


def fit_conformal(
    errors: ArrayLike,
    sigmas: ArrayLike,
    levels: Sequence[float],
    groups: ArrayLike | None = None,
    min_group: int = 0,
) -> dict[str, Any]:
    """Global and per-group conformal multipliers q_p for each level."""
    e, s = np.asarray(errors, float), np.asarray(sigmas, float)
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    scores = np.abs(e[ok]) / s[ok]
    model: dict[str, Any] = {
        "levels": list(levels),
        "n": int(ok.sum()),
        "global": {p: conformal_quantile(scores, p) for p in levels},
        "groups": {},
    }
    if groups is not None:
        g = np.asarray(groups)[ok]
        for key in np.unique(g):
            m = g == key
            if m.sum() >= min_group:
                model["groups"][key.item() if hasattr(key, "item") else key] = {
                    "n": int(m.sum()),
                    **{p: conformal_quantile(scores[m], p) for p in levels},
                }
    return model


def apply_conformal(
    sigmas: ArrayLike, model: dict[str, Any], level: float, groups: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Calibrated interval half-widths q_p * sigma (per group where fitted, else global)."""
    s = np.asarray(sigmas, float)
    q_global = model["global"][level]  # KeyError for a level that was not fitted
    q = np.full(len(s), q_global)
    if groups is not None and model["groups"]:
        g = np.asarray(groups)
        for key, gm in model["groups"].items():
            q[g == key] = gm[level]
    return q * s
