"""SynDrone loader: frames, depth, semantic labels, camera pose and 3D boxes.

Formats, read from the generation code (`LTTM/Syndrone`, `modules/carla_utils/CarlaClient.py`
and `custom_bp_tags.py`, main branch, 2026-11-04):

- Camera JSON `{x, y, z, pitch, yaw, roll}`: Unreal location (metres) and rotation
  (degrees). `z` is the height `h` above a per-frame base height `z0` that is not saved
  (recovered in `evaluate/syndrone_z0.py`).
- Depth PNG (uint16): CARLA's normalised depth `n` in [0, 1] (1000 m range) stored as
  `floor(65535 n)`, so metres = value / 65535 * 1000.
- Semantic PNG (uint8): CARLA tag per pixel (40 person, 41 rider, 100 car, 101 truck,
  102 bus, 103 train, 104 motorcycle, 105 bicycle, others for the scene).
- Box JSON: list of `{corners: 8 x [x, y, z] (Unreal world), class: [tag], id}`. The first
  entries are map ("level") boxes with ids 0, 1, 2, ...; the rest are actors with CARLA
  actor ids. Corner `i` has local signs (x, y, z) = (+,+,+), (+,+,-), (+,-,+), (+,-,-),
  (-,+,+), (-,+,-), (-,-,+), (-,-,-): odd indices are the bottom face.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerialrange.data.syndrone_archive import HEIGHTS, MemberStore, member_name
from aerialrange.geometry.camera import Camera, Pose
from aerialrange.geometry.carla import carla_camera, carla_pose

WIDTH, HEIGHT, FOV_DEG = 1920, 1080, 90.0  # CarlaClient.py `intrinsics`
FPS = 25.0  # CarlaClient.py `fps`; paper Section 3.3
N_FRAMES = 3000  # 120 s at 25 Hz (`Opt_120`), checked by scripts/check_data.py
DEPTH_MAX_M = 1000.0
BOTTOM = np.array([1, 3, 5, 7])  # corners with local z = -extent (bottom face)
TOP = np.array([0, 2, 4, 6])

PERSON, RIDER = 40, 41
VEHICLE_TAGS = (100, 101, 102, 103, 104, 105)
TAG_NAMES = {
    40: "person",
    41: "rider",
    100: "car",
    101: "truck",
    102: "bus",
    103: "train",
    104: "motorcycle",
    105: "bicycle",
}
# Camera heights and pitch (degrees) of the three SynDrone cameras (camera files; paper 3.2).
PITCH_OF_HEIGHT = {20: -30.0, 50: -60.0, 80: -90.0}


def decode_depth(png: bytes) -> NDArray[np.float32]:
    """Depth in metres from a SynDrone depth PNG."""
    a = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None or a.dtype != np.uint16 or a.ndim != 2:
        raise ValueError("depth PNG must be single-channel uint16")
    return (a.astype(np.float32) * np.float32(DEPTH_MAX_M / 65535.0)).astype(np.float32)


def encode_depth(depth_m: NDArray) -> bytes:
    """Inverse of `decode_depth` up to quantisation (for synthetic fixtures)."""
    v = np.floor(np.clip(np.asarray(depth_m) / DEPTH_MAX_M, 0.0, 1.0) * 65535.0)
    ok, buf = cv2.imencode(".png", v.astype(np.uint16))
    assert ok
    return buf.tobytes()


def decode_semantic(png: bytes) -> NDArray[np.uint8]:
    a = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None or a.dtype != np.uint8:
        raise ValueError("semantic PNG must be uint8")
    if a.ndim == 3:  # a colour PNG: tag in the red channel (CARLA convention)
        a = a[..., 2]
    return a


def decode_rgb(jpg: bytes) -> NDArray[np.uint8]:
    """BGR image (OpenCV order)."""
    a = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError("could not decode JPEG")
    return a


@dataclass(frozen=True)
class Boxes:
    """3D boxes of one frame. `corners` (N, 8, 3) in Unreal world coordinates."""

    ids: NDArray[np.int64]
    tags: NDArray[np.int64]
    corners: NDArray[np.float64]
    level: NDArray[np.bool_]  # map boxes (sequential ids), not actors

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def footprint_centre(self) -> NDArray[np.float64]:
        """Centre of the bottom face (Unreal world), (N, 3)."""
        return self.corners[:, BOTTOM].mean(axis=1)

    @property
    def centre(self) -> NDArray[np.float64]:
        return self.corners.mean(axis=1)

    def select(self, mask: NDArray[np.bool_]) -> Boxes:
        return Boxes(self.ids[mask], self.tags[mask], self.corners[mask], self.level[mask])


def parse_boxes(raw: bytes | str) -> Boxes:
    """Parse a SynDrone box file. A box's class is its first tag."""
    items = json.loads(raw)
    n = len(items)
    ids = np.array([int(b["id"]) for b in items], np.int64)
    tags = np.array([int(b["class"][0]) if b["class"] else -1 for b in items], np.int64)
    corners = np.array([b["corners"] for b in items], float).reshape(n, 8, 3)
    # Level boxes are written first with id = position (CarlaClient.py, `sbboxes`).
    k = 0
    while k < n and ids[k] == k:
        k += 1
    level = np.zeros(n, bool)
    level[:k] = True
    # An actor can be listed twice with identical content (Town10HD: one bicycle, id 122, in
    # every frame). Exact duplicates are dropped; the same id with different content is an
    # error, never silently resolved.
    keep = np.ones(n, bool)
    first: dict[int, int] = {}
    for i in np.flatnonzero(~level):
        j = first.setdefault(int(ids[i]), int(i))
        if j != i:
            if items[i] != items[j]:
                raise ValueError(f"actor id {ids[i]} listed twice with different boxes")
            keep[i] = False
    return Boxes(ids[keep], tags[keep], corners[keep], level[keep])


@dataclass(frozen=True)
class CameraRecord:
    x: float
    y: float
    h: float
    pitch: float
    yaw: float
    roll: float

    def pose(self, z0: float) -> Pose:
        return carla_pose(self.x, self.y, self.h, self.pitch, self.yaw, self.roll, z0=z0)


def parse_camera(raw: bytes | str) -> CameraRecord:
    d = json.loads(raw)
    return CameraRecord(
        float(d["x"]),
        float(d["y"]),
        float(d["z"]),
        float(d["pitch"]),
        float(d["yaw"]),
        float(d["roll"]),
    )


def syndrone_camera() -> Camera:
    return carla_camera(WIDTH, HEIGHT, FOV_DEG)


class SynDroneSequence:
    """One camera of one town: frames 0..N_FRAMES-1 at 25 Hz.

    All reads go through a `MemberStore` (local cache first, then the remote archive).
    """

    def __init__(self, store: MemberStore, town: str, height: int) -> None:
        if height not in HEIGHTS:
            raise ValueError(f"height must be one of {HEIGHTS}")
        self.store, self.town, self.height = store, town, height
        self.camera = syndrone_camera()

    @property
    def name(self) -> str:
        return f"{self.town}_h{self.height}"

    def member(self, modality: str, k: int) -> str:
        if modality == "bboxes":
            return member_name(self.town, "bboxes", k)
        return member_name(self.town, modality, k, self.height)

    def camera_record(self, k: int) -> CameraRecord:
        return parse_camera(self.store.read(self.member("camera", k)))

    def camera_records(self, frames: range | list[int] | None = None) -> list[CameraRecord]:
        frames = range(N_FRAMES) if frames is None else frames
        names = [self.member("camera", k) for k in frames]
        got = dict(self.store.iter_read(names))
        return [parse_camera(got[n]) for n in names]

    def boxes(self, k: int) -> Boxes:
        return parse_boxes(self.store.read(self.member("bboxes", k)))

    def depth(self, k: int, keep: bool = False) -> NDArray[np.float32]:
        return decode_depth(self.store.read(self.member("depth", k), keep=keep))

    def semantic(self, k: int, keep: bool = False) -> NDArray[np.uint8]:
        return decode_semantic(self.store.read(self.member("semantic", k), keep=keep))

    def rgb(self, k: int, keep: bool = False) -> NDArray[np.uint8]:
        return decode_rgb(self.store.read(self.member("rgb", k), keep=keep))
