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


def split_stratified(
    strata: Mapping[str, Sequence[str]],
    per_stratum: Mapping[str, int],
    rest: str,
    seed: int,
) -> dict[str, list[str]]:
    """Group-level split that keeps every stratum represented.

    For each stratum (in sorted order) its members are shuffled with one seeded generator; the
    first `per_stratum[split]` go to each split in the order given, and the remainder to
    `rest`. Used for SynDrone towns: whole towns, never frames, so no scene
    appears in two splits. The result does not depend on the input order.
    """
    seen: set[str] = set()
    for members in strata.values():
        if seen & set(members) or len(set(members)) != len(members):
            raise ValueError("a member appears more than once")
        seen |= set(members)
    need = sum(per_stratum.values())
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {k: [] for k in [*per_stratum, rest]}
    for name in sorted(strata):
        members = sorted(strata[name])
        if len(members) < need:
            raise ValueError(f"stratum {name!r} has {len(members)} members, needs {need}")
        shuffled = [members[i] for i in rng.permutation(len(members))]
        start = 0
        for split, n in per_stratum.items():
            out[split] += shuffled[start : start + n]
            start += n
        out[rest] += shuffled[start:]
    return {k: sorted(v) for k, v in out.items()}
