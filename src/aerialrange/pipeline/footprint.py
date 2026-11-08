"""Place the ground point at the footprint centre instead of the box's bottom edge (D-736).

With no camera roll, image rows are ground lines perpendicular to the camera's heading, so the
lowest image point of a flat footprint is its corner nearest along the camera's forward axis,
and the ground point under a box's bottom-centre lies on that corner's row. The footprint centre
is further along the camera heading by the support half-width of the footprint rectangle in that
direction,

    d(D) = (L / 2) |cos D| + (W / 2) |sin D|,

where D is the object's heading relative to the camera's. Length L and width W are random
(the group's size distribution: means, variances, covariance), and D is uncertain: it is taken
as Normal(mean, sigma) from the speed filter's heading, or uniform when there is no heading.
`forward_offset` returns the mean and variance of d over both (law of total variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

N_QUAD = 64  # quadrature points over the heading
UNIFORM_SIGMA_RAD = np.pi  # beyond this heading sigma, the heading is treated as unknown


@dataclass(frozen=True)
class FootprintSize:
    length_mean_m: float
    width_mean_m: float
    length_var_m2: float
    width_var_m2: float
    length_width_cov_m2: float

    @classmethod
    def from_mapping(cls, m: dict) -> FootprintSize:
        return cls(**{k: float(m[k]) for k in cls.__dataclass_fields__})


def _cond_moments(size: FootprintSize, d: NDArray) -> tuple[NDArray, NDArray]:
    """Mean and variance of the support half-width given relative heading(s) `d` (rad)."""
    c, s = np.abs(np.cos(d)), np.abs(np.sin(d))
    mean = 0.5 * (size.length_mean_m * c + size.width_mean_m * s)
    var = 0.25 * (
        size.length_var_m2 * c**2 + size.width_var_m2 * s**2 + 2 * size.length_width_cov_m2 * c * s
    )
    return mean, var


def forward_offset(
    size: FootprintSize, rel_heading_rad: ArrayLike, heading_sigma_rad: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Mean (m) and variance (m^2) of the distance from the bottom-edge ground point to the
    footprint centre along the camera heading, per object. NaN relative heading or a sigma
    of `UNIFORM_SIGMA_RAD` or more means the heading is unknown (uniform)."""
    mu = np.atleast_1d(np.asarray(rel_heading_rad, float))
    sg = np.broadcast_to(np.atleast_1d(np.asarray(heading_sigma_rad, float)), mu.shape)
    unknown = ~np.isfinite(mu) | ~np.isfinite(sg) | (sg >= UNIFORM_SIGMA_RAD)
    # uniform heading: midpoint rule over half a turn (d has period pi)
    du = (np.arange(N_QUAD) + 0.5) * np.pi / N_QUAD
    # normal heading: Gauss-Hermite nodes and weights
    x, w = np.polynomial.hermite_e.hermegauss(N_QUAD)
    w = w / w.sum()
    m_out = np.empty(mu.shape)
    v_out = np.empty(mu.shape)
    for i in range(mu.size):
        if unknown.flat[i]:
            d, ww = du, np.full(N_QUAD, 1.0 / N_QUAD)
        else:
            d, ww = mu.flat[i] + sg.flat[i] * x, w
        cm, cv = _cond_moments(size, d)
        m = float(ww @ cm)
        m_out.flat[i] = m
        v_out.flat[i] = float(ww @ cv + ww @ (cm - m) ** 2)
    return m_out, v_out
