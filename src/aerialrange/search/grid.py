"""A square-celled grid over the search area, in local north / east metres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid:
    """Cells of `cell_m` metres; row 0 is the southern edge (`n0_m`), column 0 the western
    edge (`e0_m`). Cell (r, c) covers north [n0 + r cell, n0 + (r + 1) cell) and east
    [e0 + c cell, e0 + (c + 1) cell). Cells are numbered r * ncols + c."""

    n0_m: float
    e0_m: float
    cell_m: float
    nrows: int
    ncols: int

    @classmethod
    def centred(cls, centre_ne: tuple[float, float], half_side_m: float, cell_m: float) -> Grid:
        n = int(np.ceil(2 * half_side_m / cell_m))
        return cls(centre_ne[0] - n * cell_m / 2, centre_ne[1] - n * cell_m / 2, cell_m, n, n)

    @property
    def size(self) -> int:
        return self.nrows * self.ncols

    @property
    def cell_area_m2(self) -> float:
        return self.cell_m * self.cell_m

    def centres(self) -> tuple[NDArray, NDArray]:
        """North and east of every cell centre, flat in cell order."""
        r, c = np.divmod(np.arange(self.size), self.ncols)
        return self.n0_m + (r + 0.5) * self.cell_m, self.e0_m + (c + 0.5) * self.cell_m

    def cell_of(self, n_m: NDArray | float, e_m: NDArray | float) -> NDArray:
        """Cell number of each point, -1 outside the grid."""
        r = np.floor((np.asarray(n_m, float) - self.n0_m) / self.cell_m).astype(np.int64)
        c = np.floor((np.asarray(e_m, float) - self.e0_m) / self.cell_m).astype(np.int64)
        ok = (r >= 0) & (r < self.nrows) & (c >= 0) & (c < self.ncols)
        return np.where(ok, r * self.ncols + c, -1)

    def cells_in_polygon(self, poly_ne: NDArray) -> NDArray:
        """Cells whose centre lies inside the polygon (k x 2, north / east), by the crossing
        rule (a centre exactly on an edge counts on one side only, so neighbouring polygons
        that share an edge never both claim a cell)."""
        poly = np.asarray(poly_ne, float)
        if len(poly) < 3:
            return np.zeros(0, np.int64)
        r0 = max(int(np.floor((poly[:, 0].min() - self.n0_m) / self.cell_m)), 0)
        r1 = min(int(np.ceil((poly[:, 0].max() - self.n0_m) / self.cell_m)), self.nrows)
        c0 = max(int(np.floor((poly[:, 1].min() - self.e0_m) / self.cell_m)), 0)
        c1 = min(int(np.ceil((poly[:, 1].max() - self.e0_m) / self.cell_m)), self.ncols)
        if r1 <= r0 or c1 <= c0:
            return np.zeros(0, np.int64)
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        pn = self.n0_m + (rr.ravel() + 0.5) * self.cell_m
        pe = self.e0_m + (cc.ravel() + 0.5) * self.cell_m
        inside = np.zeros(pn.shape, bool)
        n1, e1 = poly[:, 0], poly[:, 1]
        n2, e2 = np.roll(n1, -1), np.roll(e1, -1)
        for a_n, a_e, b_n, b_e in zip(n1, e1, n2, e2, strict=True):
            crosses = (a_n > pn) != (b_n > pn)
            with np.errstate(divide="ignore", invalid="ignore"):
                e_at = a_e + (pn - a_n) * (b_e - a_e) / (b_n - a_n)
            inside ^= crosses & (pe < e_at)
        return (rr.ravel()[inside] * self.ncols + cc.ravel()[inside]).astype(np.int64)
