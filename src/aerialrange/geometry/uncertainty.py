"""Uncertainty propagation for ground-plane ranging.

Parameters (units): u, v (pixels); h (metres, camera height above plane); roll, pitch, yaw
(radians, aerospace convention); tilt (radians, extra camera depression about body right);
fx, fy, cx, cy (pixels); obj_height (metres, used only in `centre_with_height` mode).

Noise is expressed as perturbations about the nominal inputs. `sigmas` maps parameter name to
standard deviation; parameters with no sigma (or sigma 0) are held fixed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from aerialrange.geometry.camera import Camera, Pose, pixel_to_world_ray
from aerialrange.geometry.frames import R_wb_from_ypr, rot_y, ypr_from_R_wb
from aerialrange.geometry.ranging import (
    MonteCarloSummary,
    RangeEstimate,
    estimate_range,
    estimate_range_batch,
)

PARAMS = ("u", "v", "h", "roll", "pitch", "yaw", "tilt", "fx", "fy", "cx", "cy", "obj_height")

# Central-difference step: eps^(1/3) times the parameter scale balances truncation and
# rounding error (Nocedal and Wright, Numerical Optimization, 2nd ed.).
_REL_STEP = float(np.finfo(float).eps ** (1.0 / 3.0))

# Monte Carlo percentiles: central 50, 68.27 (+/- 1 sigma), 90, 95 and 99 per cent intervals.
_ONE_SIGMA_LO = float(100 * norm.cdf(-1.0))
MC_PERCENTILES = (
    0.5,
    2.5,
    5.0,
    _ONE_SIGMA_LO,
    25.0,
    50.0,
    75.0,
    100 - _ONE_SIGMA_LO,
    95.0,
    97.5,
    99.5,
)

Correlations = Mapping[tuple[str, str], float]


def build_covariance(
    sigmas: Mapping[str, float], correlations: Correlations | None = None
) -> tuple[list[str], NDArray[np.float64]]:
    """Covariance of the active parameters (those with sigma > 0), in `PARAMS` order.

    Diagonal by default. `correlations` maps a pair of names to a correlation coefficient,
    for example {("h", "pitch"): rho} for correlated altitude and pitch.
    """
    unknown = set(sigmas) - set(PARAMS)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; expected names from {PARAMS}")
    names = [p for p in PARAMS if sigmas.get(p, 0.0) > 0]
    s = np.array([sigmas[n] for n in names], float)
    S = np.diag(s**2)
    for (a, b), rho in (correlations or {}).items():
        if a not in names or b not in names:
            raise ValueError(f"correlation ({a}, {b}) refers to a parameter with no sigma")
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"correlation ({a}, {b}) = {rho} is outside [-1, 1]")
        i, j = names.index(a), names.index(b)
        S[i, j] = S[j, i] = rho * s[i] * s[j]
    return names, S


def numerical_jacobian(
    f: Callable[[NDArray], NDArray], x: ArrayLike, steps: ArrayLike
) -> NDArray[np.float64]:
    """Central-difference Jacobian of `f` at `x` with per-parameter step sizes."""
    x = np.asarray(x, float)
    steps = np.asarray(steps, float)
    cols = []
    for i, hstep in enumerate(steps):
        e = np.zeros_like(x)
        e[i] = hstep
        cols.append((np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * hstep))
    return np.stack(cols, axis=-1)


def _nominal_values(cam: Camera, pose: Pose, uv: NDArray, h: float, obj_h: float | None) -> dict:
    yaw, pitch, roll = ypr_from_R_wb(pose.R_wb)
    return {
        "u": uv[0],
        "v": uv[1],
        "h": h,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "tilt": 0.0,
        "fx": cam.fx,
        "fy": cam.fy,
        "cx": cam.cx,
        "cy": cam.cy,
        "obj_height": obj_h or 0.0,
    }


def _model(
    cam: Camera,
    pose: Pose,
    uv: NDArray,
    h: float,
    mode: str,
    obj_h: float | None,
    names: list[str],
) -> Callable[[NDArray], NDArray]:
    """Map a perturbation vector over `names` to the ground point (NaN if invalid)."""
    yaw0, pitch0, roll0 = ypr_from_R_wb(pose.R_wb)

    def f(delta: NDArray) -> NDArray:
        d = dict.fromkeys(PARAMS, 0.0)
        d.update(zip(names, delta, strict=True))
        c = Camera.pinhole(
            cam.fx + d["fx"],
            cam.fy + d["fy"],
            cam.cx + d["cx"],
            cam.cy + d["cy"],
            R_bc=rot_y(-d["tilt"]) @ cam.R_bc,
            t_bc=cam.t_bc,
            dist=cam.dist,
        )
        p = Pose(pose.p_w, R_wb_from_ypr(yaw0 + d["yaw"], pitch0 + d["pitch"], roll0 + d["roll"]))
        o = None if obj_h is None else obj_h + d["obj_height"]
        est = estimate_range(c, p, uv + [d["u"], d["v"]], h + d["h"], mode, o)
        return est.point_w if est.valid else np.full(3, np.nan)

    return f


def ground_point_jacobian(
    cam: Camera,
    pose: Pose,
    uv: ArrayLike,
    h: float,
    sigmas: Mapping[str, float],
    mode: str = "bottom_centre",
    obj_height_m: float | None = None,
    correlations: Correlations | None = None,
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64]]:
    """Numerical Jacobian of the ground point (NED, metres) with respect to the active
    parameters. Returns (names, J of shape (3, k), parameter covariance (k, k))."""
    uv = np.asarray(uv, float)
    names, S = build_covariance(sigmas, correlations)
    nominal = _nominal_values(cam, pose, uv, h, obj_height_m)
    steps = [_REL_STEP * max(abs(nominal[n]), sigmas[n]) for n in names]
    f = _model(cam, pose, uv, h, mode, obj_height_m, names)
    J = numerical_jacobian(f, np.zeros(len(names)), steps) if names else np.zeros((3, 0))
    return names, J, S


def _derived_sigmas(P: NDArray, C: NDArray, S_P: NDArray) -> tuple[float, float, float]:
    """Range, ground-distance and azimuth (degrees) sigmas from the point covariance."""
    r_vec = P - C
    r = np.linalg.norm(r_vec)
    g_range = r_vec / r
    dn, de = r_vec[0], r_vec[1]
    gd = np.hypot(dn, de)
    # Floating-point guard: a point within rounding error of directly below has no bearing.
    if gd > np.sqrt(np.finfo(float).eps) * r:
        g_gd = np.array([dn / gd, de / gd, 0.0])
        g_az = np.array([-de / gd**2, dn / gd**2, 0.0])
        s_gd = float(np.sqrt(max(g_gd @ S_P @ g_gd, 0.0)))
        s_az = float(np.degrees(np.sqrt(max(g_az @ S_P @ g_az, 0.0))))
    else:
        s_gd = float(np.sqrt(max(np.trace(S_P[:2, :2]), 0.0)))
        s_az = float("nan")  # bearing undefined directly below the camera
    return float(np.sqrt(max(g_range @ S_P @ g_range, 0.0))), s_gd, s_az


def propagate_uncertainty(
    cam: Camera,
    pose: Pose,
    uv: ArrayLike,
    h: float,
    sigmas: Mapping[str, float],
    method: str = "first_order",
    mode: str = "bottom_centre",
    obj_height_m: float | None = None,
    correlations: Correlations | None = None,
    n_mc: int | None = None,
    rng: np.random.Generator | None = None,
) -> RangeEstimate:
    """Estimate with uncertainty. `method` is "first_order" (Sigma_P = J Sigma J^T) or
    "monte_carlo" (`n_mc` draws from `rng`, sample covariance and percentiles)."""
    if method not in ("first_order", "monte_carlo"):
        raise ValueError(f"method must be 'first_order' or 'monte_carlo', got {method!r}")
    if method == "monte_carlo" and (n_mc is None or rng is None):
        raise ValueError("monte_carlo needs n_mc and rng (n_mc is {{N_MC}} in configs)")
    uv = np.asarray(uv, float)
    nominal = estimate_range(cam, pose, uv, h, mode, obj_height_m)
    if not nominal.valid:
        return nominal
    C_w, _ = pixel_to_world_ray(cam, pose, uv)

    if method == "first_order":
        _, J, S = ground_point_jacobian(cam, pose, uv, h, sigmas, mode, obj_height_m, correlations)
        S_P = J @ S @ J.T
        s_r, s_gd, s_az = _derived_sigmas(nominal.point_w, C_w, S_P)
        return replace(
            nominal,
            cov_ne=S_P[:2, :2],
            range_sigma_m=s_r,
            ground_dist_sigma_m=s_gd,
            azimuth_sigma_deg=s_az,
        )

    names, S = build_covariance(sigmas, correlations)
    f = _model(cam, pose, uv, h, mode, obj_height_m, names)
    w, V = np.linalg.eigh(S)
    L = V * np.sqrt(np.clip(w, 0.0, None))  # works for singular (fully correlated) S
    draws = rng.standard_normal((n_mc, len(names))) @ L.T
    pts = np.array([f(dlt) for dlt in draws]) if names else np.tile(nominal.point_w, (n_mc, 1))
    ok = np.all(np.isfinite(pts), axis=1)
    pv = pts[ok]
    rv = np.linalg.norm(pv - C_w, axis=1)
    gd = np.hypot(pv[:, 0] - C_w[0], pv[:, 1] - C_w[1])
    az = np.degrees(np.arctan2(pv[:, 1] - C_w[1], pv[:, 0] - C_w[0]))
    daz = (az - nominal.azimuth_deg + 180.0) % 360.0 - 180.0
    summary = MonteCarloSummary(
        n=int(n_mc),
        n_valid=int(ok.sum()),
        range_percentiles={
            p: float(v)
            for p, v in zip(MC_PERCENTILES, np.percentile(rv, MC_PERCENTILES), strict=True)
        },
        range_sigma_m=float(np.std(rv, ddof=1)),
    )
    flags = nominal.flags + (("mc_invalid_draws",) if summary.n_valid < summary.n else ())
    return replace(
        nominal,
        cov_ne=np.cov(pv[:, :2].T),
        range_sigma_m=summary.range_sigma_m,
        ground_dist_sigma_m=float(np.std(gd, ddof=1)),
        azimuth_sigma_deg=float(np.std(daz, ddof=1)),
        flags=flags,
        mc=summary,
    )


def nonlinearity_ratio(first_order: RangeEstimate, monte_carlo: RangeEstimate) -> float:
    """Monte Carlo range sigma divided by first-order range sigma (1 in a linear regime)."""
    return monte_carlo.range_sigma_m / first_order.range_sigma_m


def reliability_flags(
    est: RangeEstimate, rel_unc_threshold: float, min_depression_deg: float
) -> tuple[str, ...]:
    """Reasons an estimate is unreliable (empty tuple means reliable).

    Thresholds are `{{REL_UNC_THRESHOLD}}` and `{{MIN_DEPRESSION_DEG}}` from configs.
    """
    if not est.valid:
        return ("invalid",)
    flags = []
    if np.isnan(est.range_sigma_m):
        flags.append("no_uncertainty")
    elif est.range_sigma_m / est.range_m > rel_unc_threshold:
        flags.append("high_relative_uncertainty")
    if est.depression_deg < min_depression_deg:
        flags.append("below_min_depression")
    return tuple(flags)


def _batch_model(
    cam: Camera, pose: Pose, uv: NDArray, h: NDArray
) -> Callable[[Mapping[str, NDArray | float]], NDArray]:
    """Ground points (N, 3) under per-sample perturbations (NaN where invalid)."""
    yaw0, pitch0, roll0 = ypr_from_R_wb(pose.R_wb)

    def f(d: Mapping[str, NDArray | float]) -> NDArray:
        g = {p: d.get(p, 0.0) for p in PARAMS}
        frame_keys = ("roll", "pitch", "yaw", "tilt", "fx", "fy", "cx", "cy")
        frame_vals = np.broadcast_arrays(
            *[np.asarray(g[p], float) for p in frame_keys], np.zeros(len(uv))
        )[:-1]
        out = np.full((len(uv), 3), np.nan)
        # Group samples that share the same per-frame perturbation (usually all or one).
        stacked = np.stack(frame_vals, axis=1)
        uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
        for j, vals in enumerate(uniq):
            v = dict(zip(frame_keys, vals, strict=True))
            idx = np.nonzero(inv.ravel() == j)[0]
            c = Camera.pinhole(
                cam.fx + v["fx"],
                cam.fy + v["fy"],
                cam.cx + v["cx"],
                cam.cy + v["cy"],
                R_bc=rot_y(-v["tilt"]) @ cam.R_bc,
                t_bc=cam.t_bc,
                dist=cam.dist,
            )
            p = Pose(
                pose.p_w, R_wb_from_ypr(yaw0 + v["yaw"], pitch0 + v["pitch"], roll0 + v["roll"])
            )
            du = np.broadcast_to(np.asarray(g["u"], float), (len(uv),))[idx]
            dv = np.broadcast_to(np.asarray(g["v"], float), (len(uv),))[idx]
            dh = np.broadcast_to(np.asarray(g["h"], float), (len(uv),))[idx]
            hh = np.broadcast_to(h, (len(uv),))[idx]
            r = estimate_range_batch(c, p, uv[idx] + np.stack([du, dv], axis=1), hh + dh)
            out[idx] = np.where(r["valid"][:, None], r["point_w"], np.nan)
        return out

    return f


def propagate_batch(
    cam: Camera,
    pose: Pose,
    uv: ArrayLike,
    h: ArrayLike,
    sigmas: Mapping[str, ArrayLike],
    correlations: Correlations | None = None,
) -> dict[str, NDArray]:
    """Vectorised first-order propagation for `bottom_centre` on a horizontal plane.

    `sigmas` values may be scalars or (N,) arrays (for example per-pixel u, v noise). Same
    step rule and outputs as `propagate_uncertainty` (tested to match). Returns the
    `estimate_range_batch` arrays plus cov_ne (N, 2, 2), range_sigma_m, ground_dist_sigma_m.
    """
    uv = np.atleast_2d(np.asarray(uv, float))
    n = len(uv)
    h = np.broadcast_to(np.asarray(h, float), (n,))
    unknown = set(sigmas) - set(PARAMS)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}")
    sig = {
        p: np.broadcast_to(np.asarray(sigmas[p], float), (n,))
        for p in PARAMS
        if p in sigmas and np.any(np.asarray(sigmas[p]) > 0)
    }
    names = list(sig)
    nominal = estimate_range_batch(cam, pose, uv, h)
    yaw, pitch, roll = ypr_from_R_wb(pose.R_wb)
    theta = {
        "u": uv[:, 0],
        "v": uv[:, 1],
        "h": h,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "tilt": 0.0,
        "fx": cam.fx,
        "fy": cam.fy,
        "cx": cam.cx,
        "cy": cam.cy,
    }
    f = _batch_model(cam, pose, uv, h)
    J = np.zeros((n, 3, len(names)))
    for i, p in enumerate(names):
        step = _REL_STEP * np.maximum(np.abs(np.broadcast_to(theta[p], (n,))), sig[p])
        if p in ("u", "v", "h"):
            J[:, :, i] = (f({p: step}) - f({p: -step})) / (2 * step[:, None])
        else:  # per-frame parameter: one shared step keeps the frame grouping intact
            s = float(np.max(step))
            J[:, :, i] = (f({p: s}) - f({p: -s})) / (2 * s)
    s_mat = np.stack([sig[p] for p in names], axis=1) if names else np.zeros((n, 0))
    R = np.eye(len(names))
    for (a, b), rho in (correlations or {}).items():
        i, j = names.index(a), names.index(b)
        R[i, j] = R[j, i] = rho
    S = s_mat[:, :, None] * R[None] * s_mat[:, None, :]
    S_P = np.einsum("nik,nkl,njl->nij", J, S, J)
    C, _ = pixel_to_world_ray(cam, pose, uv[:1])
    r_vec = nominal["point_w"] - C
    r = np.linalg.norm(r_vec, axis=1)
    g_r = r_vec / r[:, None]
    gd = np.hypot(r_vec[:, 0], r_vec[:, 1])
    g_gd = np.stack([r_vec[:, 0] / gd, r_vec[:, 1] / gd, np.zeros(n)], axis=1)
    out = dict(nominal)
    out["cov_ne"] = S_P[:, :2, :2]
    # Clamp rounding-level negatives (as the scalar version does) before the square root.
    out["range_sigma_m"] = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", g_r, S_P, g_r), 0.0))
    out["ground_dist_sigma_m"] = np.sqrt(
        np.maximum(np.einsum("ni,nij,nj->n", g_gd, S_P, g_gd), 0.0)
    )
    return out
