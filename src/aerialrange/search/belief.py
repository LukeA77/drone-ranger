"""The search map: the probability that the missing person is in each cell (Phase 14, D-752).

Bayesian search (Koopman, OEG Report 56, 1946; Stone, Theory of Optimal Search, 1975): after a
look that covers cell i with detection probability pd_i and finds nothing there, the cell's
weight is multiplied by (1 - pd_i), and the map is renormalised. Detections are not certain:
the camera also produces false alarms, at a measured rate per square metre of ground passed
over. For one stationary person and false alarms spread evenly over the covered ground
(Poisson, intensity lam per m^2), the likelihood of one observation, for "the person is in
cell i" against "the person is not under this look", is

    L_i = (1 - pd_i) + pd_i * sum_j g_j(i) / lam     for covered cells i,
    L_i = 1                                          elsewhere,

where g_j(i) is the density (per m^2) of detection j's reported position if the person stood in
cell i (the detection's position error, averaged over the cell). Every detection is either the
person or a false alarm; at most one can be the person. This is the single-target likelihood
with clutter used in probabilistic data association (Bar-Shalom and Tse, Automatica 11(5),
1975).

Observations, not maps, are shared between drones (build2 12.2). Each has a unique id (drone,
sequence number), and a map applies each id once. Applying observations is adding their log
likelihoods, so maps built from the same set of observations are equal whatever order or
route the observations arrived by, and receiving one twice changes nothing.

The map also keeps the prior probability that the person is outside the grid, which no look
changes, so the grid's total probability falls as the area is searched without a find.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from aerialrange.search.grid import Grid


@dataclass(frozen=True)
class Detection:
    """A reported person position (north, east, m) and its 2 x 2 covariance (m^2)."""

    n_m: float
    e_m: float
    cov: tuple[float, float, float, float]


@dataclass(frozen=True)
class Observation:
    """Passes that ended within one short window of one drone's flight.

    `cells` are the cells whose pass ended in the window, each at most once, with the
    probability `pd` that the camera detects a person standing there during that pass.
    `detections` are the person detections made in those passes. `fa_per_m2` is the false
    alarm intensity per square metre of ground passed over."""

    drone: int
    seq: int
    t_s: float
    cells: NDArray
    pd: NDArray
    detections: tuple[Detection, ...] = ()
    fa_per_m2: float = 0.0

    @property
    def id(self) -> tuple[int, int]:
        return (self.drone, self.seq)


def detection_density(grid: Grid, cells: NDArray, det: Detection) -> NDArray:
    """Mean density (per m^2) over each cell of the detection's position error, if the person
    stood in that cell. The error is taken along north and east with the covariance's
    diagonal; its off-diagonal term is dropped (position errors are small next to a cell)."""
    sn = np.sqrt(max(det.cov[0], 1e-12))
    se = np.sqrt(max(det.cov[3], 1e-12))
    r, c = np.divmod(cells, grid.ncols)
    n_lo = grid.n0_m + r * grid.cell_m
    e_lo = grid.e0_m + c * grid.cell_m
    pn = ndtr((n_lo + grid.cell_m - det.n_m) / sn) - ndtr((n_lo - det.n_m) / sn)
    pe = ndtr((e_lo + grid.cell_m - det.e_m) / se) - ndtr((e_lo - det.e_m) / se)
    return pn * pe / grid.cell_area_m2


TAIL_SIGMAS = 5.0  # beyond this many standard deviations a detection's density is dropped
_LL: dict[int, tuple[Observation, Grid, NDArray]] = {}


def clear_likelihood_cache() -> None:
    _LL.clear()


def log_likelihood(grid: Grid, obs: Observation) -> NDArray:
    """log L_i for the observation's cells (same order as `obs.cells`). Computed once per
    observation object and grid (every drone that receives it applies the same values)."""
    hit = _LL.get(id(obs))
    if hit is not None and hit[0] is obs and hit[1] is grid:
        return hit[2]
    ll = _log_likelihood(grid, obs)
    _LL[id(obs)] = (obs, grid, ll)
    return ll


def _log_likelihood(grid: Grid, obs: Observation) -> NDArray:
    pd = np.asarray(obs.pd, float)
    if obs.fa_per_m2 < 0:
        raise ValueError("false alarm intensity cannot be negative")
    cells = np.asarray(obs.cells, np.int64)
    s = np.zeros(len(cells))
    if obs.detections:
        order = np.argsort(cells)
        sorted_cells = cells[order]
        for d in obs.detections:
            # only cells within TAIL_SIGMAS of the detection can hold any of its density
            reach = TAIL_SIGMAS * np.sqrt(max(d.cov[0], d.cov[3], 1e-12))
            r0 = int(np.floor((d.n_m - reach - grid.n0_m) / grid.cell_m))
            r1 = int(np.floor((d.n_m + reach - grid.n0_m) / grid.cell_m))
            c0 = int(np.floor((d.e_m - reach - grid.e0_m) / grid.cell_m))
            c1 = int(np.floor((d.e_m + reach - grid.e0_m) / grid.cell_m))
            rr, cc = np.meshgrid(
                np.arange(max(r0, 0), min(r1, grid.nrows - 1) + 1),
                np.arange(max(c0, 0), min(c1, grid.ncols - 1) + 1),
                indexing="ij",
            )
            near = (rr * grid.ncols + cc).ravel()
            pos = np.searchsorted(sorted_cells, near)
            ok = pos < len(sorted_cells)
            ok[ok] = sorted_cells[pos[ok]] == near[ok]
            if ok.any():
                idx = order[pos[ok]]
                s[idx] += detection_density(grid, cells[idx], d)
    if obs.detections and obs.fa_per_m2 == 0:
        # no false alarms: a detection is the person, so L_i = pd_i * s_i up to a constant,
        # and every cell not under the look is ruled out (SearchMap.apply)
        lik = pd * s
    else:
        lik = (1.0 - pd) + (pd * s / obs.fa_per_m2 if obs.detections else 0.0)
    with np.errstate(divide="ignore"):  # a certain look (pd = 1) rules a cell out
        return np.log(lik)


@dataclass
class SearchMap:
    grid: Grid
    log_w: NDArray  # log of unnormalised cell weights
    log_outside: float  # log weight of "the person is outside the grid"
    applied: set[tuple[int, int]] = field(default_factory=set)

    @classmethod
    def from_prior(cls, grid: Grid, prior: NDArray) -> SearchMap:
        p = np.asarray(prior, float)
        if p.shape != (grid.size,) or np.any(p < 0) or p.sum() > 1 + 1e-9:
            raise ValueError("prior must be one non-negative probability per cell, sum <= 1")
        with np.errstate(divide="ignore"):
            lw = np.log(p)
        out = 1.0 - float(p.sum())
        return cls(grid, lw, float(np.log(out)) if out > 1e-15 else -np.inf)

    def apply(self, obs: Observation) -> bool:
        """Apply an observation once; returns False if it was already applied."""
        if obs.id in self.applied:
            return False
        cells = np.asarray(obs.cells, np.int64)
        if len(np.unique(cells)) != len(cells):
            raise ValueError("an observation holds each cell at most once")
        ll = log_likelihood(self.grid, obs)
        if obs.detections and obs.fa_per_m2 == 0:
            keep = self.log_w[cells] + ll
            self.log_w[:] = -np.inf
            self.log_w[cells] = keep
            self.log_outside = -np.inf
        else:
            self.log_w[cells] += ll
        self.applied.add(obs.id)
        return True

    def probabilities(self) -> tuple[NDArray, float]:
        """(probability of each cell, probability the person is outside the grid)."""
        m = max(float(self.log_w.max()), self.log_outside)
        w = np.exp(self.log_w - m)
        wo = float(np.exp(self.log_outside - m)) if np.isfinite(self.log_outside) else 0.0
        z = w.sum() + wo
        return w / z, wo / z
