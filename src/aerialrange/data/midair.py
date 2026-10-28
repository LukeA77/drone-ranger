"""Mid-Air loader.

Facts used here and their sources (Mid-Air site, https://midair.ulg.ac.be, read 2026-10-19):
- Technical specifications page: NED frames; world origin at trajectory start with yaw zero;
  cameras at 25 Hz, ground truth at 100 Hz; K with fx = cx = w/2, fy = cy = h/2; normals in
  the body frame with red -> Y, blue -> X, green -> -Z, scaled from [-1, 1] to [0, 1];
  segmentation as 8-bit class ids.
- Data organisation page: camera entry k = floor(25 t), ground-truth row floor(100 t).
- FAQ: depth is the Euclidean distance to the focal point (radial), and camera-to-body is
  (x_b, y_b, z_b) = (z_c, y_c, x_c), i.e. our level forward mount `R_BC_FORWARD`.
- Example script `hdf5-depth_viewer.py` (github.com/montefiore-ai/midair-dataset): depth PNG
  bits reinterpreted as float16; frame i uses ground-truth row 4 i.
- Example script `tools/IMU-data_generator.py`: attitude is a Hamilton quaternion (wxyz,
  pyquaternion) rotating body to world.
The HDF5 attitude dataset also carries `axis_ordering = "wxyz"`, which the loader checks.
These are re-confirmed by tests on the real data (see `aerialrange.evaluate.midair_unknowns`).
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from aerialrange.geometry.camera import Camera, Pose
from aerialrange.geometry.frames import R_BC_FORWARD, quat_to_R

CAMERA_RATE_HZ = 25.0  # technical specifications page

# Sky pixels carry the largest finite float16 value instead of infinity. Measured on
# Kite_training/sunny trajectory_0000 (2026-10-19); confirmed per frame by
# `evaluate.midair_unknowns.sky_check`.
SKY_DEPTH_M = float(np.finfo(np.float16).max)

# Segmentation ids (technical specifications page). Ground classes for geometry sampling
# are land only; water is reported separately.
GROUND_CLASSES = {
    3: "dirt_ground",
    4: "ground_vegetation",
    5: "rocky_ground",
    10: "road",
    11: "train_track",
}
WATER_CLASS = 8


def list_trajectories(h5_path: Path) -> list[str]:
    """Names of the trajectory groups in a climate's `sensor_records.hdf5`."""
    with h5py.File(h5_path, "r") as f:
        return sorted(f.keys())


def read_depth(path: Path) -> NDArray[np.float32]:
    """Depth map in metres (radial distance), decoded from float16 bits in a 16-bit PNG."""
    bits = np.asarray(Image.open(path), dtype=np.uint16)
    return bits.view(np.float16).astype(np.float32)


def sky_mask(depth_m: NDArray) -> NDArray[np.bool_]:
    """True where the depth map holds the sky sentinel."""
    return depth_m >= SKY_DEPTH_M


def read_segmentation(path: Path) -> NDArray[np.uint8]:
    """Semantic class id per pixel."""
    return np.asarray(Image.open(path), dtype=np.uint8)


def decode_normals(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Body-frame (FRD) unit normals from Mid-Air RGB normals (red Y, blue X, green -Z)."""
    c = np.asarray(rgb, float) / 255.0 * 2.0 - 1.0
    n = np.stack([c[..., 2], c[..., 0], -c[..., 1]], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def read_normals(path: Path) -> NDArray[np.float64]:
    """Body-frame unit normals from a normals PNG."""
    return decode_normals(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))


def midair_camera(width: int, height: int) -> Camera:
    """Left camera: fx = cx = w/2, fy = cy = h/2 (90 degree field of view), level forward
    mount, no lens distortion (rendered images)."""
    return Camera.pinhole(width / 2, height / 2, width / 2, height / 2, R_bc=R_BC_FORWARD)


def backproject(cam: Camera, uv: ArrayLike, depth_m: ArrayLike, kind: str) -> NDArray[np.float64]:
    """Camera-frame points (metres) for pixels `uv` (N, 2) with depth (N,).

    `kind="planar"`: depth is the distance along the optical axis, X_c = d K^-1 [u, v, 1].
    `kind="radial"`: depth is the distance to the camera centre, X_c = d unit(K^-1 [u, v, 1]).
    """
    uv = np.asarray(uv, float)
    d = np.asarray(depth_m, float)
    rays = np.stack(
        [(uv[:, 0] - cam.cx) / cam.fx, (uv[:, 1] - cam.cy) / cam.fy, np.ones(len(uv))], axis=1
    )
    if kind == "planar":
        return rays * d[:, None]
    if kind == "radial":
        return rays / np.linalg.norm(rays, axis=1, keepdims=True) * d[:, None]
    raise ValueError(f"kind must be 'planar' or 'radial', got {kind!r}")


def world_points(
    cam: Camera, pose: Pose, uv: ArrayLike, depth_m: ArrayLike, kind: str
) -> NDArray[np.float64]:
    """World points X_w = p_w + R_wb (R_bc X_c + t_bc), metres NED."""
    X_c = backproject(cam, uv, depth_m, kind)
    return pose.p_w + (X_c @ cam.R_bc.T + cam.t_bc) @ pose.R_wb.T


class MidAirTrajectory:
    """One trajectory of one climate: poses and frame paths.

    `root` is the climate folder holding `sensor_records.hdf5` and the image folders.
    """

    def __init__(self, root: Path, name: str) -> None:
        self.root = Path(root)
        self.name = name
        self._f = h5py.File(self.root / "sensor_records.hdf5", "r")
        g = self._f[name]
        att = g["groundtruth/attitude"]
        order = att.attrs.get("axis_ordering")
        order = order.decode() if isinstance(order, bytes) else order
        if order != "wxyz":
            self._f.close()
            raise ValueError(f"{name}: attitude axis_ordering is {order!r}, expected 'wxyz'")
        self._att = att[()]
        self._pos = g["groundtruth/position"][()]
        self._gt_hz = float(g["groundtruth/position"].attrs["sampling_frequency"])
        self._cam = {k: [p.decode() for p in g["camera_data"][k][()]] for k in g["camera_data"]}
        self.n_frames = len(next(iter(self._cam.values())))

    def gt_row(self, k: int) -> int:
        """Ground-truth row for camera frame k: floor(gt_rate * k / camera_rate)."""
        return math.floor(self._gt_hz * k / CAMERA_RATE_HZ)

    def pose(self, k: int) -> Pose:
        """Drone pose at camera frame k (body origin in world NED, R_wb from quaternion)."""
        r = self.gt_row(k)
        return Pose(self._pos[r], quat_to_R(self._att[r], order="wxyz"))

    def frame_path(self, modality: str, k: int) -> Path:
        """Image path for `modality` (e.g. "depth") at camera frame k."""
        return self.root / self._cam[modality][k]

    def close(self) -> None:
        self._f.close()
