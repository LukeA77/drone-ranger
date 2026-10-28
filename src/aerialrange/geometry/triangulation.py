"""Two-view triangulation with known camera poses (terrain from motion, plan D).

The drone moves, so a ground feature seen in two frames gives two viewing rays from known
positions. Their closest-approach midpoint is the feature's 3D position (metres, world NED):
no flat-ground assumption and no height above ground is needed. Uncertainty comes from pixel
noise and, optionally, pose noise, by first-order propagation (numerical Jacobian, same step
rule as `uncertainty.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerialrange.geometry.camera import Camera, Pose, pixel_to_world_ray
from aerialrange.geometry.frames import R_wb_from_ypr, ypr_from_R_wb
from aerialrange.geometry.uncertainty import _REL_STEP, numerical_jacobian

# Rays closer to parallel than this (1 - cos^2 of the angle between them) cannot be
# intersected numerically; this is a floating-point guard, not a tuned threshold. Weak but
# non-parallel geometry is handled by the covariance, not by this test.
_PARALLEL_EPS = 1e-12


def triangulate(
    cam1: Camera, pose1: Pose, uv1: ArrayLike, cam2: Camera, pose2: Pose, uv2: ArrayLike
) -> dict[str, Any]:
    """Midpoint triangulation of one feature seen at `uv1` and `uv2` (pixels).

    Returns valid, reason, point_w (metres, NED), ray_gap_m (distance between the rays at
    closest approach), parallax_deg (angle between the rays), and depth along each ray.
    """
    C1, d1 = pixel_to_world_ray(cam1, pose1, np.asarray(uv1, float))
    C2, d2 = pixel_to_world_ray(cam2, pose2, np.asarray(uv2, float))
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    b = float(d1 @ d2)
    w0 = C1 - C2
    denom = 1.0 - b * b
    nan3 = np.full(3, np.nan)
    parallax = float(np.degrees(np.arccos(np.clip(b, -1.0, 1.0))))
    if denom <= _PARALLEL_EPS:
        return {
            "valid": False,
            "reason": "parallel_rays",
            "point_w": nan3,
            "ray_gap_m": float("nan"),
            "parallax_deg": parallax,
            "depth1_m": float("nan"),
            "depth2_m": float("nan"),
        }
    dd, e = float(d1 @ w0), float(d2 @ w0)
    s = (b * e - dd) / denom
    t = (e - b * dd) / denom
    P1, P2 = C1 + s * d1, C2 + t * d2
    out = {
        "valid": s > 0 and t > 0,
        "reason": "ok" if s > 0 and t > 0 else "behind_camera",
        "point_w": (P1 + P2) / 2,
        "ray_gap_m": float(np.linalg.norm(P1 - P2)),
        "parallax_deg": parallax,
        "depth1_m": float(s),
        "depth2_m": float(t),
    }
    if not out["valid"]:
        out["point_w"] = nan3
    return out


def _perturbed_pose(pose: Pose, dpos: NDArray, datt: NDArray) -> Pose:
    yaw, pitch, roll = ypr_from_R_wb(pose.R_wb)
    return Pose(pose.p_w + dpos, R_wb_from_ypr(yaw + datt[0], pitch + datt[1], roll + datt[2]))


def triangulate_with_covariance(
    cam1: Camera,
    pose1: Pose,
    uv1: ArrayLike,
    cam2: Camera,
    pose2: Pose,
    uv2: ArrayLike,
    sigma_px: float,
    sigma_pos_m: float = 0.0,
    sigma_att_rad: float = 0.0,
) -> dict[str, Any]:
    """`triangulate` plus a first-order 3x3 covariance `cov_w` (m^2) of the point.

    Noise: `sigma_px` on each pixel coordinate in both images; optionally `sigma_pos_m` on
    each camera position axis and `sigma_att_rad` on each of yaw, pitch and roll, per frame
    (independent between the two frames).
    """
    uv1, uv2 = np.asarray(uv1, float), np.asarray(uv2, float)
    base = triangulate(cam1, pose1, uv1, cam2, pose2, uv2)
    if not base["valid"]:
        base["cov_w"] = np.full((3, 3), np.nan)
        return base
    sig = np.array([sigma_px] * 4 + [sigma_pos_m] * 6 + [sigma_att_rad] * 6)
    active = sig > 0

    def f(delta_active: NDArray) -> NDArray:
        dl = np.zeros(len(sig))
        dl[active] = delta_active
        p1 = _perturbed_pose(pose1, dl[4:7], dl[10:13])
        p2 = _perturbed_pose(pose2, dl[7:10], dl[13:16])
        r = triangulate(cam1, p1, uv1 + dl[0:2], cam2, p2, uv2 + dl[2:4])
        return r["point_w"]

    nominal = np.concatenate([uv1, uv2, np.zeros(12)])
    steps = _REL_STEP * np.maximum(np.abs(nominal[active]), sig[active])
    J = numerical_jacobian(f, np.zeros(int(active.sum())), steps)
    base["cov_w"] = J @ np.diag(sig[active] ** 2) @ J.T
    return base


def _midpoint_batch(
    C1: NDArray, D1: NDArray, C2: NDArray, D2: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Vectorised closest-approach midpoint for N ray pairs (unit directions)."""
    b = np.einsum("ij,ij->i", D1, D2)
    w0 = C1 - C2
    denom = 1.0 - b * b
    dd = D1 @ w0 if w0.ndim == 1 else np.einsum("ij,ij->i", D1, w0)
    e = D2 @ w0 if w0.ndim == 1 else np.einsum("ij,ij->i", D2, w0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b * e - dd) / denom
        t = (e - b * dd) / denom
    P1 = C1 + s[:, None] * D1
    P2 = C2 + t[:, None] * D2
    ok = (denom > _PARALLEL_EPS) & (s > 0) & (t > 0)
    return (P1 + P2) / 2, ok, np.linalg.norm(P1 - P2, axis=1), b, denom


def triangulate_batch(
    cam1: Camera,
    pose1: Pose,
    uv1: ArrayLike,
    cam2: Camera,
    pose2: Pose,
    uv2: ArrayLike,
    sigma_px: float,
    sigma_pos_m: float = 0.0,
    sigma_att_rad: float = 0.0,
) -> dict[str, NDArray]:
    """Vectorised `triangulate_with_covariance` for N features between the same two frames.

    Pixel noise is independent per feature; pose noise is shared by all features of a frame.
    Same step rule as the single-point version (tested to match).
    """
    uv1 = np.atleast_2d(np.asarray(uv1, float))
    uv2 = np.atleast_2d(np.asarray(uv2, float))
    n = len(uv1)

    def solve(u1: NDArray, u2: NDArray, p1: Pose, p2: Pose) -> tuple[NDArray, ...]:
        C1, D1 = pixel_to_world_ray(cam1, p1, u1)
        C2, D2 = pixel_to_world_ray(cam2, p2, u2)
        D1 = D1 / np.linalg.norm(D1, axis=1, keepdims=True)
        D2 = D2 / np.linalg.norm(D2, axis=1, keepdims=True)
        return _midpoint_batch(C1, D1, C2, D2)

    P, ok, gap, b, _ = solve(uv1, uv2, pose1, pose2)
    cov = np.zeros((n, 3, 3))
    # Pixel parameters: per feature, independent, so each column is perturbed for all at once.
    if sigma_px > 0:
        for j, (arr, col) in enumerate(((uv1, 0), (uv1, 1), (uv2, 0), (uv2, 1))):
            step = _REL_STEP * np.maximum(np.abs(arr[:, col]), sigma_px)
            plus_1, minus_1 = uv1.copy(), uv1.copy()
            plus_2, minus_2 = uv2.copy(), uv2.copy()
            target_p, target_m = (plus_1, minus_1) if j < 2 else (plus_2, minus_2)
            target_p[:, col] += step
            target_m[:, col] -= step
            Pp = solve(plus_1, plus_2, pose1, pose2)[0]
            Pm = solve(minus_1, minus_2, pose1, pose2)[0]
            g = (Pp - Pm) / (2 * step[:, None])
            cov += sigma_px**2 * np.einsum("ni,nj->nij", g, g)
    # Pose parameters: shared by all features of a frame.
    for which in (0, 1):
        for kind, sig in (("pos", sigma_pos_m), ("att", sigma_att_rad)):
            if sig <= 0:
                continue
            for axis in range(3):
                h = _REL_STEP * sig
                grads = []
                for sign in (1.0, -1.0):
                    dv = np.zeros(3)
                    dv[axis] = sign * h
                    dpos, datt = (dv, np.zeros(3)) if kind == "pos" else (np.zeros(3), dv)
                    p1 = _perturbed_pose(pose1, dpos, datt) if which == 0 else pose1
                    p2 = _perturbed_pose(pose2, dpos, datt) if which == 1 else pose2
                    grads.append(solve(uv1, uv2, p1, p2)[0])
                g = (grads[0] - grads[1]) / (2 * h)
                cov += sig**2 * np.einsum("ni,nj->nij", g, g)
    cov[~ok] = np.nan
    P = np.where(ok[:, None], P, np.nan)
    return {
        "valid": ok,
        "point_w": P,
        "cov_w": cov,
        "ray_gap_m": gap,
        "parallax_deg": np.degrees(np.arccos(np.clip(b, -1.0, 1.0))),
    }
