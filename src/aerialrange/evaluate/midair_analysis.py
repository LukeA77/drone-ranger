"""Mid-Air analysis from cached samples. No images are read.

The cache (written by `midair_ranging.evaluate_trajectory`) stores per sample: pixel, image
size, drone pose, true world point and range, patch-averaged world normal, and the oracle and
relative-altitude plane heights. Everything here is recomputed from those, so noise sweeps,
Monte Carlo, sensitivity and the slope experiments need no second pass over the images.

Bases (`base`): "oracle_plane" or "relative_altitude"; the noisy-inputs mode
adds sensor noise on top of either.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerialrange.data.midair import midair_camera
from aerialrange.geometry.camera import Camera, Pose, pixel_to_world_ray
from aerialrange.geometry.frames import R_wb_from_ypr, quat_to_R, rot_y, ypr_from_R_wb
from aerialrange.geometry.ranging import estimate_range_batch
from aerialrange.geometry.uncertainty import PARAMS, propagate_batch

BASES = {"oracle_plane": "z_oracle", "relative_altitude": "z_rel_ref"}
FRAME_PARAMS = ("h", "roll", "pitch", "yaw", "tilt", "fx", "fy", "cx", "cy")
PIXEL_PARAMS = ("u", "v")


def _plane_key(base: str) -> str:
    if base not in BASES:
        raise ValueError(f"base must be one of {sorted(BASES)}, got {base!r}")
    return BASES[base]


def frames(
    r: Mapping[str, NDArray], idx: NDArray | None = None
) -> Iterator[tuple[NDArray, Camera, Pose]]:
    """Yield (sample indices, camera, pose) per (group, frame), optionally within `idx`."""
    sel = np.arange(len(r["u"])) if idx is None else np.asarray(idx)
    keys = np.stack([r["group"][sel], r["frame"][sel]], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    for j in range(len(uniq)):
        ii = sel[inv.ravel() == j]
        i0 = ii[0]
        cam = midair_camera(int(r["img_w"][i0]), int(r["img_h"][i0]))
        pose = Pose(r["pose_p"][i0], quat_to_R(r["pose_q"][i0]))
        yield ii, cam, pose


def _perturbed(cam: Camera, pose: Pose, d: Mapping[str, float]) -> tuple[Camera, Pose]:
    yaw, pitch, roll = ypr_from_R_wb(pose.R_wb)
    c = Camera.pinhole(
        cam.fx + d.get("fx", 0.0),
        cam.fy + d.get("fy", 0.0),
        cam.cx + d.get("cx", 0.0),
        cam.cy + d.get("cy", 0.0),
        R_bc=rot_y(-d.get("tilt", 0.0)) @ cam.R_bc,
        t_bc=cam.t_bc,
        dist=cam.dist,
    )
    p = Pose(
        pose.p_w,
        R_wb_from_ypr(
            yaw + d.get("yaw", 0.0), pitch + d.get("pitch", 0.0), roll + d.get("roll", 0.0)
        ),
    )
    return c, p


def _height(r: Mapping[str, NDArray], ii: NDArray, cam: Camera, pose: Pose, base: str) -> NDArray:
    C_z = (pose.p_w + pose.R_wb @ cam.t_bc)[2]
    return r[_plane_key(base)][ii] - C_z


def _errors(
    r: Mapping[str, NDArray], ii: NDArray, P: NDArray, rng_m: NDArray, valid: NDArray
) -> dict[str, NDArray]:
    signed = rng_m - r["true_range_m"][ii]
    return {
        "valid": valid,
        "range_est_m": rng_m,
        "range_err_m": signed,
        "rel_range_err": signed / r["true_range_m"][ii],
        "horiz_err_m": np.hypot(*(P[:, :2] - r["true_point_w"][ii, :2]).T),
    }


def _assemble(n: int, parts: list[tuple[NDArray, dict[str, NDArray]]]) -> dict[str, NDArray]:
    keys = parts[0][1].keys()
    out = {}
    for k in keys:
        shape = (n,) + parts[0][1][k].shape[1:]
        arr = np.zeros(shape, bool) if parts[0][1][k].dtype == bool else np.full(shape, np.nan)
        for ii, d in parts:
            arr[ii] = d[k]
        out[k] = arr
    return out


def noisy_mode(
    r: Mapping[str, NDArray], sigmas: Mapping[str, float], base: str, rng: np.random.Generator
) -> dict[str, NDArray]:
    """`noisy_inputs` mode: the estimator sees inputs perturbed by Gaussian noise with
    `sigmas` (per frame: h, attitude, tilt, intrinsics; per pixel: u, v), and propagates the
    same `sigmas` from the inputs it sees. Empty `sigmas` gives the noise-free base mode."""
    _plane_key(base)
    unknown = set(sigmas) - set(PARAMS)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}")
    parts = []
    for ii, cam, pose in frames(r):
        d = {p: float(rng.normal(0, sigmas[p])) for p in FRAME_PARAMS if sigmas.get(p, 0) > 0}
        c, p = _perturbed(cam, pose, d)
        uv = np.stack([r["u"][ii], r["v"][ii]], axis=1).astype(float)
        for j, q in enumerate(PIXEL_PARAMS):
            if sigmas.get(q, 0) > 0:
                uv[:, j] += rng.normal(0, sigmas[q], len(ii))
        h = _height(r, ii, cam, pose, base) + d.get("h", 0.0)
        if any(v > 0 for v in sigmas.values()):
            est = propagate_batch(c, p, uv, h, sigmas)
            sig = est["range_sigma_m"]
            cov = est["cov_ne"]
        else:
            est = estimate_range_batch(c, p, uv, h)
            sig = np.zeros(len(ii))
            cov = np.zeros((len(ii), 2, 2))
        e = _errors(r, ii, est["point_w"], est["range_m"], est["valid"])
        e.update(
            pred_range_sigma_m=np.where(est["valid"], sig, np.nan),
            pred_cov_ne=cov,
            depression_est_deg=est["depression_deg"],
        )
        parts.append((ii, e))
    return _assemble(len(r["u"]), parts)


def first_order_sigma(
    r: Mapping[str, NDArray], idx: NDArray, sigmas: Mapping[str, float], base: str
) -> NDArray:
    """Predicted first-order range sigma at the noise-free inputs, for samples `idx`."""
    out = np.full(len(r["u"]), np.nan)
    for ii, cam, pose in frames(r, idx):
        uv = np.stack([r["u"][ii], r["v"][ii]], axis=1).astype(float)
        out[ii] = propagate_batch(cam, pose, uv, _height(r, ii, cam, pose, base), sigmas)[
            "range_sigma_m"
        ]
    return out[np.asarray(idx)]


def monte_carlo_sigma(
    r: Mapping[str, NDArray],
    idx: NDArray,
    sigmas: Mapping[str, float],
    base: str,
    n_mc: int,
    rng: np.random.Generator,
) -> NDArray:
    """Monte Carlo range sigma (`n_mc` draws, `{{N_MC}}`) about the noise-free inputs.
    Invalid draws are dropped per sample; NaN if fewer than two valid draws."""
    out = np.full(len(r["u"]), np.nan)
    for ii, cam, pose in frames(r, idx):
        uv0 = np.stack([r["u"][ii], r["v"][ii]], axis=1).astype(float)
        h0 = _height(r, ii, cam, pose, base)
        ranges = np.full((n_mc, len(ii)), np.nan)
        for b in range(n_mc):
            d = {p: float(rng.normal(0, sigmas[p])) for p in FRAME_PARAMS if sigmas.get(p, 0) > 0}
            c, p = _perturbed(cam, pose, d)
            uv = uv0.copy()
            for j, q in enumerate(PIXEL_PARAMS):
                if sigmas.get(q, 0) > 0:
                    uv[:, j] += rng.normal(0, sigmas[q], len(ii))
            est = estimate_range_batch(c, p, uv, h0 + d.get("h", 0.0))
            ranges[b] = np.where(est["valid"], est["range_m"], np.nan)
        ok = np.sum(np.isfinite(ranges), axis=0) >= 2
        out[ii[ok]] = np.nanstd(ranges[:, ok], axis=0, ddof=1)
    return out[np.asarray(idx)]


def sensitivity(
    r: Mapping[str, NDArray], sigmas: Mapping[str, float], base: str
) -> dict[str, dict[str, Any]]:
    """First-order range-sigma contribution of each input alone (tornado chart).
    Contributions add in quadrature to the total because the parameter covariance is diagonal."""
    idx = np.arange(len(r["u"]))
    out = {}
    for p, s in sigmas.items():
        per = first_order_sigma(r, idx, {p: s}, base)
        out[p] = {
            "sigma": s,
            "per_sample": per,
            "median_m": float(np.nanmedian(per)),
            "p90_m": float(np.nanpercentile(per, 90)),
        }
    return out


def _intersect(
    cam: Camera, pose: Pose, uv: NDArray, n: NDArray, off: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Rays through `uv` against planes n_i . X = off_i (one plane per sample)."""
    C, d = pixel_to_world_ray(cam, pose, uv)
    nd = np.einsum("ij,ij->i", n, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (off - n @ C) / nd
    valid = np.isfinite(lam) & (lam > 0) & (np.abs(nd) > np.finfo(float).eps)
    lam = np.where(valid, lam, np.nan)
    P = C + lam[:, None] * d
    return P, lam * np.linalg.norm(d, axis=1), valid


def robust_plane(X: NDArray, n_iter: int = 3, k_mad: float = 3.0) -> tuple[NDArray, float]:
    """Plane n . X = off fitted by least squares with residual trimming at `k_mad` scaled
    MADs (a standard robust rule; iterations and k are documented in D-211). n points up.
    Needs at least 3 points; trimming never leaves fewer than 3."""
    if len(X) < 3:
        raise ValueError("a plane needs at least 3 points")
    keep = np.ones(len(X), bool)
    for _ in range(n_iter):
        c = X[keep].mean(axis=0)
        n = np.linalg.svd(X[keep] - c, full_matrices=False)[2][2]
        res = (X - c) @ n
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        if mad == 0:
            break
        new_keep = np.abs(res) <= k_mad * mad
        if new_keep.sum() < 3:
            break
        keep = new_keep
    if n[2] > 0:
        n = -n
    return n, float(n @ c)


def tilted_plane_mode(r: Mapping[str, NDArray]) -> dict[str, NDArray]:
    """Diagnostic (H4): one robust tilted plane per frame through the true ground points."""
    parts = []
    for ii, cam, pose in frames(r):
        if len(ii) < 3:  # no plane from fewer than 3 points: invalid, not dropped
            nan = np.full(len(ii), np.nan)
            parts.append(
                (ii, _errors(r, ii, np.full((len(ii), 3), np.nan), nan, np.zeros(len(ii), bool)))
            )
            continue
        n, off = robust_plane(r["true_point_w"][ii])
        uv = np.stack([r["u"][ii], r["v"][ii]], axis=1).astype(float)
        P, rng_m, valid = _intersect(cam, pose, uv, np.tile(n, (len(ii), 1)), np.full(len(ii), off))
        parts.append((ii, _errors(r, ii, P, rng_m, valid)))
    return _assemble(len(r["u"]), parts)


def local_slope_mode(r: Mapping[str, NDArray]) -> dict[str, NDArray]:
    """Diagnostic (H4): a plane per sample with the local (patch-averaged) true normal,
    anchored at the frame medoid (the sampled true ground point nearest the coordinate-wise
    median), so the anchor lies on the terrain."""
    parts = []
    for ii, cam, pose in frames(r):
        X = r["true_point_w"][ii]
        anchor = X[np.argmin(np.linalg.norm(X - np.median(X, axis=0), axis=1))]
        n = r["normal_w"][ii]
        uv = np.stack([r["u"][ii], r["v"][ii]], axis=1).astype(float)
        P, rng_m, valid = _intersect(cam, pose, uv, n, n @ anchor)
        parts.append((ii, _errors(r, ii, P, rng_m, valid)))
    return _assemble(len(r["u"]), parts)
