"""Trajectory-level splits.

Mid-Air is split by trajectory, never by frame, so correlated frames of one trajectory stay
together. A trajectory name maps to the same split in every climate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def allocate_counts(n: int, proportions: Mapping[str, float]) -> dict[str, int]:
    """Integer counts per split by the largest-remainder method. Proportions must sum to 1."""
    if not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"proportions must sum to 1, got {sum(proportions.values())}")
    raw = {k: n * p for k, p in proportions.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    left = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:left]:
        counts[k] += 1
    return counts


def split_trajectories(
    names: Sequence[str], proportions: Mapping[str, float], seed: int
) -> dict[str, list[str]]:
    """Assign trajectory names to splits with a seeded shuffle (independent of input order)."""
    ordered = sorted(names)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in perm]
    out, start = {}, 0
    for k, c in allocate_counts(len(ordered), proportions).items():
        out[k] = sorted(shuffled[start : start + c])
        start += c
    return out
