"""Mid-Air ranging accuracy.

For sampled ground pixels the ground truth comes from depth (radial, D-201); estimates come
from the deployed horizontal-plane ranging. Modes:
- `oracle_plane`: horizontal plane at the robust (median) Down coordinate of the frame's
  sampled ground points. Upper bound; remaining error is terrain not being a plane.
- `relative_altitude`: plane fixed at the first usable frame's oracle height, so height
  above it follows the drone's true vertical motion (a take-off-relative altimeter).
The `noisy_inputs` mode adds sensor noise on top (see `noisy_estimates`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_dilation, uniform_filter

from aerialrange.data.midair import (
    GROUND_CLASSES,
    WATER_CLASS,
    MidAirTrajectory,
    decode_normals,
    midair_camera,
    read_depth,
    read_segmentation,
    sky_mask,
    world_points,
)
from aerialrange.geometry.camera import Camera, Pose
from aerialrange.geometry.frames import R_to_quat
from aerialrange.geometry.ranging import estimate_range_batch

SAMPLE_CLASSES = np.array(sorted(GROUND_CLASSES) + [WATER_CLASS])


@dataclass(frozen=True)
class Sampling:
    """Pixel sampling settings (`{{PIX_PER_FRAME}}`, `{{FRAME_STRIDE}}`, `{{EDGE_MARGIN_PX}}`,
    `{{DEPTH_EDGE_REL}}`, `{{NORMAL_PATCH_PX}}` in configs/data.yaml)."""

    pix_per_frame: int
    frame_stride: int
    edge_margin_px: int
    depth_edge_rel: float
    normal_patch_px: int


def discontinuity_mask(depth: NDArray, seg: NDArray, rel: float) -> NDArray[np.bool_]:
    """Pixels next to a depth jump larger than `rel` (relative), sky, or a non-sample class."""
    bad = sky_mask(depth) | ~np.isin(seg, SAMPLE_CLASSES)
    edge = np.zeros_like(bad)
    for axis in (0, 1):
        a = np.take(depth, range(depth.shape[axis] - 1), axis=axis)
        b = np.take(depth, range(1, depth.shape[axis]), axis=axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            jump = np.abs(a - b) / np.minimum(a, b) > rel
        sl_a = [slice(None)] * 2
        sl_b = [slice(None)] * 2
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        edge[tuple(sl_a)] |= jump
        edge[tuple(sl_b)] |= jump
    return edge | bad


def candidate_mask(depth: NDArray, seg: NDArray, s: Sampling) -> NDArray[np.bool_]:
    """Sample-class pixels at least `edge_margin_px` from any discontinuity or image border."""
    disc = discontinuity_mask(depth, seg, s.depth_edge_rel)
    m = s.edge_margin_px
    if m > 0:
        disc = binary_dilation(disc, structure=np.ones((2 * m + 1, 2 * m + 1), bool))
    ok = np.isin(seg, SAMPLE_CLASSES) & ~disc
    if m > 0:
        ok[:m] = ok[-m:] = False
        ok[:, :m] = ok[:, -m:] = False
    return ok


def patch_normal_world(
    normals_rgb: NDArray,
    ground_lowres: NDArray,
    R_wb: NDArray,
    u: NDArray,
    v: NDArray,
    scale: int,
    patch: int,
) -> NDArray:
    """World-frame unit normals (NED, pointing up) from body-frame normals averaged over a
    square patch of `patch` normals pixels, using only sample-class pixels (D-206)."""
    n = decode_normals(normals_rgb) * ground_lowres[..., None]
    mean = np.stack(
        [uniform_filter(n[..., i], size=patch, mode="nearest") for i in range(3)], axis=-1
    )
    m = mean[v // scale, u // scale] @ R_wb.T
    with np.errstate(invalid="ignore"):
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def slope_deg(normal_w: NDArray) -> NDArray:
    """Angle of a world normal from vertical, in degrees (0 for level ground)."""
    return np.degrees(np.arccos(np.clip(-normal_w[:, 2], -1, 1)))


def ground_truth(cam: Camera, pose: Pose, uv: NDArray, depth: NDArray) -> dict[str, NDArray]:
    """True world points and range for pixels with radial depth."""
    X = world_points(cam, pose, uv, depth, "radial")
    C = pose.p_w + pose.R_wb @ cam.t_bc
    rv = X - C
    return {
        "point_w": X,
        "range_m": np.linalg.norm(rv, axis=1),
        "ground_dist_m": np.hypot(rv[:, 0], rv[:, 1]),
        "depression_deg": np.degrees(np.arctan2(rv[:, 2], np.hypot(rv[:, 0], rv[:, 1]))),
    }


def errors(est: dict[str, NDArray], truth: dict[str, NDArray]) -> dict[str, NDArray]:
    """Signed and relative range error and horizontal position error (metres)."""
    signed = est["range_m"] - truth["range_m"]
    return {
        "valid": est["valid"],
        "range_est_m": est["range_m"],
        "range_err_m": signed,
        "rel_range_err": signed / truth["range_m"],
        "horiz_err_m": np.hypot(*(est["point_w"][:, :2] - truth["point_w"][:, :2]).T),
    }


def evaluate_frame(
    traj: MidAirTrajectory,
    k: int,
    s: Sampling,
    rng: np.random.Generator,
    z_rel_ref: float | None,
) -> dict[str, Any] | None:
    """Sample one frame. Returns per-sample arrays, or None if it has no candidate pixels.

    `z_rel_ref` is the relative-altitude plane (Down, metres); None makes this frame's oracle
    plane the reference (returned in `z_rel_ref`).
    """
    depth = read_depth(traj.frame_path("depth", k))
    seg = read_segmentation(traj.frame_path("segmentation", k))
    ok = candidate_mask(depth, seg, s)
    vv, uu = np.nonzero(ok)
    if len(uu) == 0:
        return None
    pick = rng.choice(len(uu), size=min(s.pix_per_frame, len(uu)), replace=False)
    u, v = uu[pick], vv[pick]
    uv = np.stack([u, v], axis=1).astype(float)
    cam = midair_camera(depth.shape[1], depth.shape[0])
    pose = traj.pose(k)
    truth = ground_truth(cam, pose, uv, depth[v, u].astype(float))
    C_z = float((pose.p_w + pose.R_wb @ cam.t_bc)[2])
    z_oracle = float(np.median(truth["point_w"][:, 2]))
    nrm = np.asarray(Image.open(traj.frame_path("normals", k)).convert("RGB"))
    scale = depth.shape[0] // nrm.shape[0]
    ground_lowres = ok[::scale, ::scale][: nrm.shape[0], : nrm.shape[1]]
    normal_w = patch_normal_world(nrm, ground_lowres, pose.R_wb, u, v, scale, s.normal_patch_px)
    n = len(u)
    out: dict[str, Any] = {
        "frame": np.full(n, k),
        "img_w": np.full(n, depth.shape[1]),
        "img_h": np.full(n, depth.shape[0]),
        "pose_p": np.tile(pose.p_w, (n, 1)),
        "pose_q": np.tile(R_to_quat(pose.R_wb), (n, 1)),
        "true_point_w": truth["point_w"],
        "normal_w": normal_w,
        "u": u,
        "v": v,
        "cls": seg[v, u],
        "true_range_m": truth["range_m"],
        "true_ground_dist_m": truth["ground_dist_m"],
        "true_depression_deg": truth["depression_deg"],
        "slope_deg": slope_deg(normal_w),
        "z_oracle": np.full(len(u), z_oracle),
        "h_oracle_m": np.full(len(u), z_oracle - C_z),
    }
    # The first usable frame of a trajectory defines the relative-altitude reference.
    z_rel = z_oracle if z_rel_ref is None else z_rel_ref
    out["z_rel_ref"] = np.full(len(u), z_rel)
    modes = {"oracle_plane": z_oracle, "relative_altitude": z_rel}
    for mode, z in modes.items():
        e = errors(estimate_range_batch(cam, pose, uv, z - C_z), truth)
        out.update({f"{mode}.{key}": val for key, val in e.items()})
    return out


def evaluate_trajectory(root: Path, name: str, s: Sampling, seed: int) -> dict[str, NDArray]:
    """All sampled frames of one trajectory, concatenated. The relative-altitude reference is
    the oracle plane of the first frame with candidate pixels."""
    rng = np.random.default_rng([seed, int(name.split("_")[-1])])
    traj = MidAirTrajectory(root, name)
    parts: list[dict[str, Any]] = []
    z_ref = None
    try:
        for k in range(0, traj.n_frames, s.frame_stride):
            r = evaluate_frame(traj, k, s, rng, z_ref)
            if r is None:
                continue
            z_ref = float(r["z_rel_ref"][0])
            parts.append(r)
    finally:
        traj.close()
    keys = sorted(set().union(*parts)) if parts else []
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def sampling_dict(s: Sampling) -> dict[str, Any]:
    return asdict(s)
