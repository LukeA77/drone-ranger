"""Confirmed finds and recurring false alarms (Phase 16, D-764).

Confirmation: a crew can act on a place only once it has been seen more than once. An alarm is
raised at a detection when, counting it, at least `k` detections from the whole team lie within
`radius_m` of it (each detection in the simulation ends a separate pass, so these are separate
looks). One place raises one alarm: a detection within `radius_m` of an alarm already raised adds
nothing. An alarm within `radius_m` of the person is the confirmed find; the others are false
alarms sent to the crew.

Recurring false alarms, two kinds, fitted to SynDrone by maximum likelihood
(scripts/multidrone_clutter_fit.py): a share of false alarms comes from persistent look-alikes,
fixed things on the ground that fire with a fitted probability on each pass over them; the rest
are one-off. The look-alikes are placed once per search, uniformly, at the density that gives the
measured share of the measured rate (density x fire probability = share x rate), so ground
flown again and again does not collect new ones. One-off false alarms come at the measured rate
times (1 - share), so the total stays the measured rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerialrange.search.grid import Grid


@dataclass
class Alarms:
    k: int
    radius_m: float
    cell_m: float
    pts: list[tuple[float, float]] = field(default_factory=list)
    _cells: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    raised: list[tuple[float, float, float, bool]] = field(default_factory=list)  # n, e, t, true
    _raised_cells: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k < 1 or self.radius_m <= 0 or self.cell_m < self.radius_m:
            raise ValueError("need k >= 1 and 0 < radius <= cell size")

    def _key(self, n: float, e: float) -> tuple[int, int]:
        return int(np.floor(n / self.cell_m)), int(np.floor(e / self.cell_m))

    def _near(self, idx: dict, pts: list, n: float, e: float) -> int:
        a, b = self._key(n, e)
        c = 0
        for i in (a - 1, a, a + 1):
            for j in (b - 1, b, b + 1):
                for p in idx.get((i, j), ()):
                    pn, pe = pts[p][0], pts[p][1]
                    if (pn - n) ** 2 + (pe - e) ** 2 <= self.radius_m**2:
                        c += 1
        return c

    def add(self, n: float, e: float, t: float, person: tuple[float, float]) -> bool | None:
        """Record a detection; returns None if no alarm is raised, else whether the alarm is
        the person (within the radius of them)."""
        self._cells.setdefault(self._key(n, e), []).append(len(self.pts))
        self.pts.append((n, e))
        if self._near(self._raised_cells, self.raised, n, e):
            return None
        if self._near(self._cells, self.pts, n, e) < self.k:
            return None
        true = (n - person[0]) ** 2 + (e - person[1]) ** 2 <= self.radius_m**2
        self._raised_cells.setdefault(self._key(n, e), []).append(len(self.raised))
        self.raised.append((n, e, t, bool(true)))
        return bool(true)


@dataclass
class ClutterSites:
    """Persistent look-alikes by grid cell; each fires with probability `fire` on each pass."""

    grid: Grid
    fire_p: float
    sites: dict[int, list[tuple[float, float]]] = field(default_factory=dict)

    def add(self, n: float, e: float) -> None:
        self.sites.setdefault(int(self.grid.cell_of(n, e)), []).append((n, e))

    def place(self, density_per_m2: float, rng: np.random.Generator) -> int:
        """Place look-alikes uniformly over the grid (Poisson count); returns how many."""
        if density_per_m2 <= 0:
            return 0
        g = self.grid
        k = int(rng.poisson(density_per_m2 * g.size * g.cell_area_m2))
        n = g.n0_m + rng.random(k) * g.nrows * g.cell_m
        e = g.e0_m + rng.random(k) * g.ncols * g.cell_m
        for a, b in zip(n.tolist(), e.tolist(), strict=True):
            self.add(a, b)
        return k

    def fire(
        self, ended: np.ndarray, sigma_m: float, rng: np.random.Generator
    ) -> list[tuple[float, float]]:
        """Detections of look-alikes in the cells whose pass just ended."""
        if self.fire_p <= 0 or not self.sites:
            return []
        out = []
        for c in ended.tolist():
            for n, e in self.sites.get(c, ()):
                if rng.random() < self.fire_p:
                    out.append((n + rng.normal(0.0, sigma_m), e + rng.normal(0.0, sigma_m)))
        return out
