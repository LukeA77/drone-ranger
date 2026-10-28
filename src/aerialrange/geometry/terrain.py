"""Local terrain surface from triangulated ground points, and ranging against it (plan D).

Instead of assuming one flat ground plane, the ground near the target is modelled as a local
plane height field, Down = a + b (north - n0) + c (east - e0), fitted robustly to 3D points
triangulated from the drone's own motion. The target's viewing ray is intersected with that
surface, iterating because the fit is centred where the ray meets the ground.

Uncertainty is measured per scene, not assumed:
  - how well the surface height at the target is pinned down by the points (fit covariance);
  - local roughness: the scatter of ground points about the fitted surface beyond their own
    measurement noise, because the target's own patch of ground deviates by about that much.
Range sigma follows from the height sigma: d(range)/d(height) = 1 / (d_z - b d_n - c d_e) for a
unit ray direction d.
Units: metres, world NED. Outlier trimming uses the same documented rule as D-211 (3 scaled
MADs, up to 3 passes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MAD_SCALE = 1.4826  # MAD to standard deviation for Gaussian data


def _wls(A: NDArray, z: NDArray, w: NDArray) -> tuple[NDArray, NDArray]:
    Aw = A * w[:, None]
    M = A.T @ Aw
    p = np.linalg.solve(M, Aw.T @ z)
    return p, np.linalg.inv(M)


def fit_local_surface(
    points: ArrayLike,
    center_ne: ArrayLike,
    radius_m: float | None,
    min_points: int,
    point_var_z: ArrayLike | None = None,
    k_mad: float = 3.0,
    n_iter: int = 3,
    k_nearest: int | None = None,
) -> dict[str, Any]:
    """Robust local plane height field around `center_ne` from points within `radius_m`
    (horizontal), or from the `k_nearest` points horizontally (adapts to point density; used
    when given). `point_var_z`: optional per-point measurement variance of Down (m^2).

    Returns valid, reason, params (a, b, c) in local coordinates, cov (3x3), roughness_m,
    height_at_center, height_var_at_center, n_support, support_radius_m.
    """
    P = np.asarray(points, float)
    c0 = np.asarray(center_ne, float)
    dist = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
    if k_nearest is not None:
        sel = np.zeros(len(P), bool)
        if len(P):
            sel[np.argsort(dist, kind="stable")[: min(k_nearest, len(P))]] = True
    elif radius_m is not None:
        sel = dist <= radius_m
    else:
        raise ValueError("give radius_m or k_nearest")
    out: dict[str, Any] = {"valid": False, "reason": "too_few_points", "n_support": int(sel.sum())}
    if sel.sum() < max(min_points, 3):
        return out
    X = P[sel, 0] - c0[0]
    Y = P[sel, 1] - c0[1]
    Z = P[sel, 2]
    var_m = np.zeros(len(Z)) if point_var_z is None else np.asarray(point_var_z, float)[sel]
    A = np.column_stack([np.ones(len(Z)), X, Y])
    keep = np.ones(len(Z), bool)
    rough2 = 0.0
    for _ in range(n_iter):
        w = 1.0 / (var_m[keep] + rough2 + 1e-12)
        p, _ = _wls(A[keep], Z[keep], w)
        r = Z - A @ p
        rk = r[keep]
        dof = max(int(keep.sum()) - 3, 1)
        rough2 = max(float(np.sum(rk**2) / dof - np.mean(var_m[keep])), 0.0)
        mad = _MAD_SCALE * np.median(np.abs(rk - np.median(rk)))
        if mad == 0:
            break
        new_keep = np.abs(r) <= k_mad * mad
        if new_keep.sum() < max(min_points, 3) or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= max(min_points, 3) else keep
            break
        keep = new_keep
    w = 1.0 / (var_m[keep] + rough2 + 1e-12)
    p, cov = _wls(A[keep], Z[keep], w)
    if np.all(var_m == 0):
        # No per-point variances: weights are 1 / roughness^2, so cov is already scaled.
        pass
    out.update(
        valid=True,
        reason="ok",
        params=p,
        cov=cov,
        roughness_m=float(np.sqrt(rough2)),
        height_at_center=float(p[0]),
        height_var_at_center=float(cov[0, 0]),
        n_support=int(keep.sum()),
        center_ne=c0,
        support_radius_m=float(dist[sel].max()),
    )
    return out


def range_on_local_terrain(
    origin: ArrayLike,
    direction: ArrayLike,
    points: ArrayLike,
    radius_m: float | None,
    min_points: int,
    point_var_z: ArrayLike | None = None,
    n_iter: int = 6,
    tol_m: float = 1e-6,
    k_nearest: int | None = None,
    init_center_ne: ArrayLike | None = None,
) -> dict[str, Any]:
    """Intersect the ray `origin + lam * direction` with the local terrain surface.

    `init_center_ne` (north, east, metres) starts the search there, for example where an
    image-neighbour estimate put the target; otherwise it starts where the ray meets the
    median point height.

    Returns valid, reason, point_w, range_m, range_sigma_m (surface fit plus local roughness),
    height_sigma_m, roughness_m, slope_deg, n_support.
    """
    C = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    P = np.asarray(points, float)
    nan = float("nan")
    fail = {
        "valid": False,
        "point_w": np.full(3, np.nan),
        "range_m": nan,
        "range_sigma_m": nan,
        "height_sigma_m": nan,
        "roughness_m": nan,
        "slope_deg": nan,
        "n_support": 0,
    }
    if len(P) == 0:
        return {**fail, "reason": "too_few_points"}
    # Start where the ray meets the median ground height, else under the point nearest the ray.
    zmed = float(np.median(P[:, 2]))
    if init_center_ne is not None:
        center = np.asarray(init_center_ne, float)
    elif d[2] > 0 and zmed > C[2]:
        center = (C + (zmed - C[2]) / d[2] * d)[:2]
    else:
        t = np.clip((P - C) @ d, 0, None)
        center = P[np.argmin(np.linalg.norm(P - (C + t[:, None] * d), axis=1)), :2]
    surf: dict[str, Any] = {}
    lam = nan
    for _ in range(n_iter):
        surf = fit_local_surface(P, center, radius_m, min_points, point_var_z, k_nearest=k_nearest)
        if not surf["valid"]:
            return {**fail, "reason": surf["reason"], "n_support": surf["n_support"]}
        a, b, c = surf["params"]
        denom = d[2] - b * d[0] - c * d[1]
        rhs = a + b * (C[0] - center[0]) + c * (C[1] - center[1]) - C[2]
        if abs(denom) < 1e-12 or rhs / denom <= 0:
            return {**fail, "reason": "no_intersection", "n_support": surf["n_support"]}
        lam = rhs / denom
        new_center = (C + lam * d)[:2]
        moved = float(np.linalg.norm(new_center - center))
        center = new_center
        if moved < tol_m:
            break
    # Final fit centred on the intersection: prediction variance there is cov[0, 0].
    surf = fit_local_surface(P, center, radius_m, min_points, point_var_z, k_nearest=k_nearest)
    if not surf["valid"]:
        return {**fail, "reason": surf["reason"], "n_support": surf["n_support"]}
    a, b, c = surf["params"]
    denom = d[2] - b * d[0] - c * d[1]
    lam = (a + b * (C[0] - center[0]) + c * (C[1] - center[1]) - C[2]) / denom
    if lam <= 0:
        return {**fail, "reason": "no_intersection", "n_support": surf["n_support"]}
    h_var = surf["height_var_at_center"] + surf["roughness_m"] ** 2
    return {
        "valid": True,
        "reason": "ok",
        "point_w": C + lam * d,
        "range_m": float(lam),
        "range_sigma_m": float(np.sqrt(h_var) / abs(denom)),
        "height_sigma_m": float(np.sqrt(h_var)),
        "roughness_m": surf["roughness_m"],
        "slope_deg": float(np.degrees(np.arctan(np.hypot(b, c)))),
        "n_support": surf["n_support"],
        "support_radius_m": surf["support_radius_m"],
    }
