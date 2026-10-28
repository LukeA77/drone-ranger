"""Terrain from the drone's own motion, evaluated on Mid-Air (plan D).

For an evaluated frame k: track corner features from frame k back to frame k - gap (pyramidal
Lucas-Kanade with a forward-backward consistency check), triangulate them with the two known
camera poses, then range each target pixel against the local terrain surface fitted to the
k nearest triangulated points (`geometry.terrain`).

Deployable by construction: it uses only images and poses. It does not use Mid-Air's
segmentation or depth, which serve only as ground truth for scoring. Settings are
`{{TERRAIN_*}}` placeholders chosen on the development split (docs/decisions.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from aerialrange.data.midair import MidAirTrajectory, midair_camera
from aerialrange.geometry.camera import Camera, Pose, pixel_to_world_ray
from aerialrange.geometry.terrain import range_on_local_terrain
from aerialrange.geometry.triangulation import triangulate_batch


@dataclass(frozen=True)
class MotionSettings:
    frame_gaps: tuple[int, ...]  # earlier frames tried per feature (per-point baseline choice)
    max_features: int  # corners detected in frame k
    quality: float  # Shi-Tomasi quality level (relative to the strongest corner)
    min_distance_px: int  # minimum spacing between corners
    fb_max_px: float  # forward-backward tracking consistency limit
    sigma_px_by_gap: tuple[float, ...]  # measured tracking noise per frame gap (pixels)
    k_nearest: int  # points per local terrain fit
    min_support: int  # minimum points for a fit
    neighbours: str = "world"  # "world": k nearest in 3D around the ray; "image": k nearest
    # features to the target pixel in frame k (same visible surface patch); "image_then_world":
    # image neighbours locate the patch, then a world k-nearest fit refines there (D-405)
    sigma_pos_m: float = 0.0  # pose position noise (per axis) for the covariance
    sigma_att_rad: float = 0.0  # pose attitude noise (per angle) for the covariance


def detect_features(img_k: NDArray, s: MotionSettings) -> NDArray:
    """Shi-Tomasi corners in frame k, shape (N, 1, 2) float32 (OpenCV layout)."""
    p0 = cv2.goodFeaturesToTrack(
        img_k, maxCorners=s.max_features, qualityLevel=s.quality, minDistance=s.min_distance_px
    )
    return np.zeros((0, 1, 2), np.float32) if p0 is None else p0


def track_to(
    img_k: NDArray, img_prev: NDArray, p0: NDArray, s: MotionSettings
) -> tuple[NDArray, NDArray]:
    """Positions of the frame-k corners `p0` in an earlier frame, and a success mask
    (Lucas-Kanade both ways, forward-backward error below `fb_max_px`)."""
    if len(p0) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(img_k, img_prev, p0, None)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(img_prev, img_k, p1, None)
    fb = np.linalg.norm(p0 - pb, axis=2).ravel()
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < s.fb_max_px)
    return p1[:, 0].astype(float), ok


def track_frame(
    img_k: NDArray, prev_imgs: list[NDArray], s: MotionSettings
) -> tuple[NDArray, list[tuple[NDArray, NDArray]]]:
    """Frame-k corners and, for each earlier image, their tracked positions and success mask.
    Independent of the poses, so it can be reused across pose perturbations (D-621)."""
    p0 = detect_features(img_k, s)
    return p0[:, 0].astype(float), [track_to(img_k, img_p, p0, s) for img_p in prev_imgs]


def multi_baseline_points(
    cam: Camera,
    img_k: NDArray,
    pose_k: Pose,
    prev: list[tuple[NDArray, Pose, float]],
    s: MotionSettings,
) -> tuple[NDArray, NDArray, NDArray]:
    """Triangulate each frame-k corner against every earlier frame where it tracks, and keep,
    per corner, the estimate with the smallest total position variance. Near ground (fast
    image motion) ends up with short baselines, far ground with long ones."""
    uvk, tracks = track_frame(img_k, [img for img, _, _ in prev], s)
    return points_from_tracks(cam, pose_k, uvk, tracks, [(pose, sig) for _, pose, sig in prev], s)


def points_from_tracks(
    cam: Camera,
    pose_k: Pose,
    uvk: NDArray,
    tracks: list[tuple[NDArray, NDArray]],
    prev_poses: list[tuple[Pose, float]],
    s: MotionSettings,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The triangulation and per-corner baseline choice of `multi_baseline_points`, from
    tracks already computed by `track_frame` and one pose (and pixel sigma) per earlier frame."""
    n = len(uvk)
    best_P = np.full((n, 3), np.nan)
    best_var = np.full(n, np.nan)
    best_tr = np.full(n, np.inf)
    best_par = np.full(n, np.nan)
    for (uvp, ok), (pose_p, sigma_px) in zip(tracks, prev_poses, strict=True):
        if not ok.any():
            continue
        P, var_z, par, valid, trace = ground_points(
            cam, pose_k, pose_p, uvk[ok], uvp[ok], s, sigma_px
        )
        idx = np.nonzero(ok)[0][valid]
        # Select by total 3D variance: for shallow rays the error lies mostly along the
        # nearly horizontal line of sight, which the Down variance alone hides.
        better = trace < best_tr[idx]
        best_P[idx[better]] = P[better]
        best_var[idx[better]] = var_z[better]
        best_tr[idx[better]] = trace[better]
        best_par[idx[better]] = par[better]
    keep = np.isfinite(best_tr)
    return best_P[keep], best_var[keep], best_par[keep], uvk[keep]


def ground_points(
    cam: Camera,
    pose_k: Pose,
    pose_prev: Pose,
    uv_k: NDArray,
    uv_prev: NDArray,
    s: MotionSettings,
    sigma_px: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Triangulated 3D points (NED, metres), Down variance, parallax (degrees), the validity
    mask over the input features, and the total position variance (trace of the covariance,
    m^2). All but the mask are for valid features only."""
    if len(uv_k) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0)
    r = triangulate_batch(
        cam,
        pose_k,
        uv_k,
        cam,
        pose_prev,
        uv_prev,
        sigma_px=sigma_px,
        sigma_pos_m=s.sigma_pos_m,
        sigma_att_rad=s.sigma_att_rad,
    )
    v = r["valid"] & np.isfinite(r["cov_w"][:, 2, 2])
    trace = np.trace(r["cov_w"][v], axis1=1, axis2=2)
    return r["point_w"][v], r["cov_w"][v, 2, 2], r["parallax_deg"][v], v, trace


def range_targets(
    cam: Camera,
    pose: Pose,
    uv: NDArray,
    P: NDArray,
    var_z: NDArray,
    s: MotionSettings,
    uv_points: NDArray | None = None,
) -> dict[str, NDArray]:
    """Terrain-aware range for each target pixel against the local surface.

    With `s.neighbours == "image"`, the surface is fitted to the k triangulated features nearest
    to the target pixel in the image (`uv_points`), i.e. the patch of ground the camera actually
    sees around the target; otherwise to the k points nearest in 3D around the ray."""
    n = len(uv)
    keys = ("range_m", "range_sigma_m", "roughness_m", "slope_deg", "support_radius_m")
    out: dict[str, NDArray] = {k: np.full(n, np.nan) for k in keys}
    out["valid"] = np.zeros(n, bool)
    out["n_support"] = np.zeros(n, int)
    out["point_w"] = np.full((n, 3), np.nan)
    if len(P) < s.min_support:
        return out
    C, D = pixel_to_world_ray(cam, pose, uv)
    for i in range(n):
        if s.neighbours in ("image", "image_then_world"):
            if uv_points is None:
                raise ValueError("image neighbours need the feature pixels")
            d_img = np.hypot(*(uv_points - uv[i]).T)
            idx = np.argsort(d_img, kind="stable")[: s.k_nearest]
            r = range_on_local_terrain(
                C,
                D[i],
                P[idx],
                radius_m=np.inf,
                min_points=s.min_support,
                point_var_z=var_z[idx],
            )
            if s.neighbours == "image_then_world" and r["valid"]:
                # Refine around the visible patch found in the image (D-405).
                r2 = range_on_local_terrain(
                    C,
                    D[i],
                    P,
                    radius_m=None,
                    min_points=s.min_support,
                    point_var_z=var_z,
                    k_nearest=s.k_nearest,
                    init_center_ne=r["point_w"][:2],
                )
                if r2["valid"]:
                    r = r2
        else:
            r = range_on_local_terrain(
                C,
                D[i],
                P,
                radius_m=None,
                min_points=s.min_support,
                point_var_z=var_z,
                k_nearest=s.k_nearest,
            )
        out["valid"][i] = r["valid"]
        out["n_support"][i] = r["n_support"]
        out["point_w"][i] = r["point_w"]
        for k in keys:
            out[k][i] = r.get(k, np.nan)
    return out


def evaluate_trajectory_motion(
    traj: MidAirTrajectory, samples: dict[str, NDArray], s: MotionSettings
) -> dict[str, Any]:
    """Terrain-from-motion estimates for every cached sample of one trajectory.

    `samples` is the trajectory's cache from `midair_ranging.evaluate_trajectory` (frame,
    u, v); results are returned in the same sample order.
    """
    n = len(samples["u"])
    res: dict[str, Any] = {"terrain.valid": np.zeros(n, bool)}
    for k in ("range_m", "range_sigma_m", "roughness_m", "slope_deg", "support_radius_m"):
        res[f"terrain.{k}"] = np.full(n, np.nan)
    res["terrain.n_support"] = np.zeros(n, int)
    res["terrain.point_w"] = np.full((n, 3), np.nan)
    res["terrain.n_points"] = np.zeros(n, int)
    res["terrain.median_parallax_deg"] = np.full(n, np.nan)
    for k in np.unique(samples["frame"]):
        idx = np.nonzero(samples["frame"] == k)[0]
        gaps = [g for g in s.frame_gaps if k - g >= 0]
        if not gaps:
            continue
        img_k = cv2.imread(str(traj.frame_path("color_left", int(k))), cv2.IMREAD_GRAYSCALE)
        cam = midair_camera(img_k.shape[1], img_k.shape[0])
        pose_k = traj.pose(int(k))
        sig = dict(zip(s.frame_gaps, s.sigma_px_by_gap, strict=True))
        prev = [
            (
                cv2.imread(str(traj.frame_path("color_left", int(k) - g)), cv2.IMREAD_GRAYSCALE),
                traj.pose(int(k) - g),
                sig[g],
            )
            for g in gaps
        ]
        P, var_z, par, uv_pts = multi_baseline_points(cam, img_k, pose_k, prev, s)
        uv = np.stack([samples["u"][idx], samples["v"][idx]], axis=1).astype(float)
        out = range_targets(cam, pose_k, uv, P, var_z, s, uv_points=uv_pts)
        for key, val in out.items():
            res[f"terrain.{key}"][idx] = val
        res["terrain.n_points"][idx] = len(P)
        res["terrain.median_parallax_deg"][idx] = np.median(par) if len(par) else np.nan
    return res


def settings_dict(s: MotionSettings) -> dict[str, Any]:
    return asdict(s)
