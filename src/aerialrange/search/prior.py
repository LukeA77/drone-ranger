"""Where the missing person may be, before any search (Phase 14, D-752).

Distance-ring model (Syrotuck; Koester, Lost Person Behavior, 2008): circles around the initial
planning point (IPP) holding 25, 50, 75 and 95 % of past finds of a subject category. As in the
ring model evaluated by Sava, Twardy, Koester and Sonwalkar (Transactions in GIS 20(1), 2016):
each ring's probability is spread evenly over its area, and the remaining 5 % evenly over the
rest of a 50 x 50 km region around the IPP.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerialrange.search.grid import Grid

# Hiker, temperate mountainous terrain, n = 568: horizontal distance from the IPP, km, at the
# 25 / 50 / 75 / 95 % points (Koester, Lost Person Behavior Instructor Activity Guide, 2010,
# "Scenario Weighing" table, p. 103). Sava et al. (2016, Table 1) list the same case as 0.7,
# 1.9, 3.6 and 11.3 labelled km; those are the same distances in miles (x 1.609 gives 1.13,
# 3.06, 5.79, 18.2), so Koester's own kilometre values are used.
HIKER_TEMPERATE_MOUNTAIN_KM = (1.1, 3.1, 5.8, 18.3)
RING_SHARES = (0.25, 0.25, 0.25, 0.20)
OUTSIDE_SHARE = 0.05
OUTSIDE_REGION_M = 50_000.0  # side of the square around the IPP that holds the last 5 %


def ring_density_per_m2(dist_m: NDArray, radii_m: tuple[float, ...]) -> NDArray:
    """Prior density (per square metre) at distance `dist_m` from the IPP."""
    r = np.asarray(radii_m, float)
    if len(r) != len(RING_SHARES) or np.any(np.diff(r) <= 0) or r[0] <= 0:
        raise ValueError("ring radii must be four increasing positive distances")
    inner = np.r_[0.0, r[:-1]]
    area = np.pi * (r**2 - inner**2)
    dens = np.asarray(RING_SHARES) / area
    outside = OUTSIDE_SHARE / (OUTSIDE_REGION_M**2 - np.pi * r[-1] ** 2)
    k = np.searchsorted(r, np.asarray(dist_m, float), side="right")
    return np.where(k < len(r), dens[np.minimum(k, len(r) - 1)], outside)


def ring_prior(grid: Grid, ipp_ne: tuple[float, float], radii_m: tuple[float, ...]) -> NDArray:
    """Probability of each cell (cell centre's density times cell area). The sum is the
    prior probability that the person is inside the grid at all."""
    n, e = grid.centres()
    d = np.hypot(n - ipp_ne[0], e - ipp_ne[1])
    return ring_density_per_m2(d, radii_m) * grid.cell_area_m2


def uniform_prior(grid: Grid) -> NDArray:
    """Every cell equally likely (Koopman's classic case), holding probability 1."""
    return np.full(grid.size, 1.0 / grid.size)


def sample_location(prior: NDArray, grid: Grid, rng: np.random.Generator) -> tuple[float, float]:
    """A person's position drawn from the prior, given that they are inside the grid: a cell
    with probability proportional to its prior, then a uniform point in it."""
    p = prior / prior.sum()
    k = int(rng.choice(grid.size, p=p))
    r, c = divmod(k, grid.ncols)
    u, v = rng.random(2)
    return grid.n0_m + (r + u) * grid.cell_m, grid.e0_m + (c + v) * grid.cell_m
