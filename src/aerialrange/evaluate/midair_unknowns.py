"""Resolve the Mid-Air unknowns by test on real data.

Checks (each returns plain numbers for `results/midair/unknowns.json`):
1. `quaternion_check`: IMU specific force against R^T (a_w - g) for both R and R^T, with
   gravity along +Down. The convention with the smaller residual is the dataset's.
2. `water_plane_check`: water (class 8) is genuinely flat. Back-project water pixels with
   planar and radial depth, fit a plane, and compare residuals. The plane normal rotated to
   the world frame should be vertical if the attitude and camera mount are right.
3. `normals_check`: decoded body-frame normals of water pixels rotated to the world frame
   should also be vertical (checks the normals decoding). Reports the tilt of the frame-mean
   normal (decoding bias) and the per-pixel median tilt (per-pixel noise).
4. `sky_check`: sky sentinel depth against segmentation class 0.
5. `frame_mapping_check`: ground-truth length against 4 x number of frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from aerialrange.data.midair import (
    WATER_CLASS,
    MidAirTrajectory,
    decode_normals,
    midair_camera,
    read_depth,
    read_segmentation,
    sky_mask,
    world_points,
)
from aerialrange.geometry.frames import quat_to_R

# Gravity used by the dataset's own IMU generator script (tools/IMU-data_generator.py).
MIDAIR_GRAVITY_MS2 = 9.81


def fit_plane(X: NDArray) -> tuple[NDArray, float]:
    """Least-squares plane through points X (N, 3): unit normal and RMS residual (metres)."""
    c = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - c, full_matrices=False)
    return vt[2], float(s[2] / np.sqrt(len(X)))


def _angle_from_vertical_deg(n: NDArray) -> float:
    return float(np.degrees(np.arccos(min(1.0, abs(n[2]) / np.linalg.norm(n)))))


def quaternion_check(h5_path: Path, name: str) -> dict[str, float]:
    """RMS of IMU accelerometer minus predicted specific force, per candidate convention."""
    with h5py.File(h5_path, "r") as f:
        g = f[name]
        q = g["groundtruth/attitude"][()]
        a_w = g["groundtruth/acceleration"][()]
        acc = g["imu/accelerometer"][()] - g["imu/accelerometer"].attrs["init_bias_est"]
    sf_w = a_w - np.array([0.0, 0.0, MIDAIR_GRAVITY_MS2])  # specific force, gravity along +Down
    out = {}
    for label, transpose in (("R_wb_body_to_world", True), ("R_bw_world_to_body", False)):
        pred = np.array(
            [
                (quat_to_R(qi).T if transpose else quat_to_R(qi)) @ s
                for qi, s in zip(q, sf_w, strict=True)
            ]
        )
        out[label] = float(np.sqrt(np.mean(np.sum((acc - pred) ** 2, axis=1))))
    return out


def _frames(traj: MidAirTrajectory, stride: int) -> range:
    return range(0, traj.n_frames, stride)


def water_plane_check(
    root: Path, name: str, stride: int, min_pixels: int, max_pixels: int, seed: int
) -> list[dict[str, Any]]:
    """Per-frame plane fit to water pixels for planar and radial depth."""
    rng = np.random.default_rng(seed)
    traj = MidAirTrajectory(root, name)
    rows = []
    try:
        for k in _frames(traj, stride):
            seg = read_segmentation(traj.frame_path("segmentation", k))
            vv, uu = np.nonzero(seg == WATER_CLASS)
            if len(uu) < min_pixels:
                continue
            pick = rng.choice(len(uu), size=min(max_pixels, len(uu)), replace=False)
            uv = np.stack([uu[pick], vv[pick]], axis=1).astype(float)
            depth = read_depth(traj.frame_path("depth", k))[vv[pick], uu[pick]]
            cam = midair_camera(seg.shape[1], seg.shape[0])
            pose = traj.pose(k)
            row: dict[str, Any] = {
                "trajectory": name,
                "frame": k,
                "n_pixels": len(pick),
                "median_depth_m": float(np.median(depth)),
            }
            for kind in ("planar", "radial"):
                n_w, rms = fit_plane(world_points(cam, pose, uv, depth, kind))
                row[f"{kind}_rms_m"] = rms
                row[f"{kind}_tilt_deg"] = _angle_from_vertical_deg(n_w)
            rows.append(row)
    finally:
        traj.close()
    return rows


def normals_check(root: Path, name: str, stride: int, min_pixels: int) -> list[dict[str, Any]]:
    """Angle from vertical of decoded water normals rotated into the world frame."""
    traj = MidAirTrajectory(root, name)
    rows = []
    try:
        for k in _frames(traj, stride):
            rgb = np.asarray(Image.open(traj.frame_path("normals", k)).convert("RGB"))
            seg = read_segmentation(traj.frame_path("segmentation", k))
            # Normals are stored at a lower resolution; sample segmentation at matching pixels.
            sy, sx = seg.shape[0] // rgb.shape[0], seg.shape[1] // rgb.shape[1]
            water = seg[::sy, ::sx][: rgb.shape[0], : rgb.shape[1]] == WATER_CLASS
            if water.sum() < min_pixels:
                continue
            n_w = decode_normals(rgb[water]) @ traj.pose(k).R_wb.T
            tilt = np.degrees(np.arccos(np.clip(-n_w[:, 2], -1, 1)))
            mean_n = n_w.mean(axis=0)
            rows.append(
                {
                    "trajectory": name,
                    "frame": k,
                    "n_pixels": int(water.sum()),
                    "mean_normal_tilt_deg": _angle_from_vertical_deg(mean_n),
                    "median_tilt_deg": float(np.median(tilt)),
                    "p95_tilt_deg": float(np.percentile(tilt, 95)),
                    "resolution_ratio": [sy, sx],
                }
            )
    finally:
        traj.close()
    return rows


def sky_check(root: Path, name: str, stride: int) -> dict[str, Any]:
    """Agreement between the sky depth sentinel and segmentation class 0."""
    traj = MidAirTrajectory(root, name)
    both = sky_only = seg0_only = total = 0
    try:
        for k in _frames(traj, stride):
            s = sky_mask(read_depth(traj.frame_path("depth", k)))
            z = read_segmentation(traj.frame_path("segmentation", k)) == 0
            both += int((s & z).sum())
            sky_only += int((s & ~z).sum())
            seg0_only += int((~s & z).sum())
            total += s.size
    finally:
        traj.close()
    return {
        "trajectory": name,
        "pixels": total,
        "sky_and_class0": both,
        "sky_not_class0": sky_only,
        "class0_not_sky": seg0_only,
    }


def frame_mapping_check(h5_path: Path, name: str) -> dict[str, int]:
    """Number of camera frames and ground-truth rows."""
    with h5py.File(h5_path, "r") as f:
        g = f[name]
        return {"frames": len(g["camera_data/depth"]), "gt_rows": len(g["groundtruth/position"])}
