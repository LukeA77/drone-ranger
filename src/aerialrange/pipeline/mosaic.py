"""Ground image of a flight, built from its own camera frames (map background, D-750).

Each frame is laid onto flat ground with the frame's camera pose: for a pinhole camera over a
plane, the map from image pixels to ground points is a homography, so four pixel-to-ground
correspondences (from `evaluate.auair_world.ground_offset_oblique`, the same ray model the
pipeline ranges with) fix it exactly, and `cv2.warpPerspective` resamples the frame onto a north-up
grid. Only image rows whose line of sight is at least `min_dep_deg` below the horizon are used,
the rows the map counts as searched ground; nearer the horizon the ground is stretched beyond
use. Where frames overlap, each grid cell keeps the frame whose camera was nearest to it on the
ground (the steepest, least stretched view of that cell).

The result is a picture for orientation, not a measured orthophoto: it assumes flat ground at
the height the pipeline assumes, so raised objects (buildings, trees, vehicles) lean away from
the camera, and any error in the recorded pose moves the picture as it moves the ranges.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerialrange.evaluate.auair_world import ground_offset_oblique

DEFAULT_RES_M = 0.1  # finest grid cell; coarsened if the area would exceed MAX_PX
MAX_PX = 4096  # longest side of the ground image


@dataclass(frozen=True)
class Camera:
    focal_px: float
    cx: float
    cy: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class GroundImage:
    image: NDArray  # BGR, north up, east right
    covered: NDArray  # bool, the cells some frame filled
    north_max_m: float  # local north of the top edge
    east_min_m: float  # local east of the left edge
    res_m: float  # grid cell size
    frames_used: int

    @property
    def north_min_m(self) -> float:
        return self.north_max_m - self.image.shape[0] * self.res_m

    @property
    def east_max_m(self) -> float:
        return self.east_min_m + self.image.shape[1] * self.res_m


def usable_top_row(cam: Camera, tilt_deg: float, min_dep_deg: float) -> float:
    """Topmost image row (on the centre column) seen at least `min_dep_deg` below the horizon,
    clipped to the image; equal to the image height when no row qualifies."""
    v = cam.cy + cam.focal_px * np.tan(np.radians(min_dep_deg - tilt_deg))
    return float(np.clip(v, 0.0, cam.height_px))


def frame_quad(pose: pd.Series, cam: Camera, min_dep_deg: float) -> tuple[NDArray, NDArray]:
    """Image corners of the usable part of a frame (4 x 2: u, v) and their ground points
    (4 x 2: north, east, local metres); empty arrays when nothing usable is in view."""
    tilt = float(pose["tilt_deg"])
    if not (np.isfinite(tilt) and np.isfinite(pose["h_m"]) and pose["h_m"] > 0):
        return np.zeros((0, 2)), np.zeros((0, 2))
    top = usable_top_row(cam, tilt, min_dep_deg)
    if top >= cam.height_px:
        return np.zeros((0, 2)), np.zeros((0, 2))
    w, h = float(cam.width_px), float(cam.height_px)
    uv = np.array([[0.0, top], [w, top], [w, h], [0.0, h]])
    dn, de = ground_offset_oblique(
        uv[:, 0],
        uv[:, 1],
        float(pose["h_m"]),
        cam.focal_px,
        float(pose["yaw_rad"]),
        tilt,
        cam.cx,
        cam.cy,
    )
    ne = np.c_[dn + float(pose["n_m"]), de + float(pose["e_m"])]
    if not np.isfinite(ne).all():
        return np.zeros((0, 2)), np.zeros((0, 2))
    return uv, ne


def build_ground_image(
    frames: Iterable[tuple[int, NDArray]],
    flight: pd.DataFrame,
    cam: Camera,
    min_dep_deg: float,
    res_m: float = DEFAULT_RES_M,
    max_px: int = MAX_PX,
) -> GroundImage | None:
    """Ground image from (frame number, BGR image) pairs and the flight's poses (`flight`
    indexed by frame, with n_m, e_m, h_m, yaw_rad, tilt_deg). Frames without a usable view are
    skipped. Returns None when no frame has one."""
    quads = {}
    for f, pose in flight.iterrows():
        uv, ne = frame_quad(pose, cam, min_dep_deg)
        if len(ne):
            quads[int(f)] = (uv, ne)
    if not quads:
        return None
    allne = np.vstack([q[1] for q in quads.values()])
    n_max, e_min = float(allne[:, 0].max()), float(allne[:, 1].min())
    span = max(float(allne[:, 0].max() - allne[:, 0].min()), float(allne[:, 1].max() - e_min))
    res = max(res_m, span / (max_px - 2))
    rows = int(np.ceil((n_max - allne[:, 0].min()) / res)) + 2
    cols = int(np.ceil((allne[:, 1].max() - e_min) / res)) + 2
    canvas = np.zeros((rows, cols, 3), np.uint8)
    best = np.full((rows, cols), np.inf, np.float32)
    used = 0
    for f, img in frames:
        if f not in quads:
            continue
        uv, ne = quads[f]
        pose = flight.loc[f]
        # grid position (column = east, row = south) of the quad and of the point below the drone
        px = np.c_[(ne[:, 1] - e_min) / res, (n_max - ne[:, 0]) / res]
        x0, y0 = np.floor(px.min(axis=0)).astype(int)
        x1, y1 = np.ceil(px.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cols), min(y1, rows)
        if x1 <= x0 or y1 <= y0:
            continue
        H = cv2.getPerspectiveTransform(uv.astype(np.float32), (px - [x0, y0]).astype(np.float32))
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpPerspective(img, H, size, flags=cv2.INTER_LINEAR)
        valid = np.zeros(img.shape[:2], np.uint8)
        valid[int(np.ceil(uv[0, 1])) :, :] = 1
        mask = cv2.warpPerspective(valid, H, size, flags=cv2.INTER_NEAREST).astype(bool)
        cn = (float(pose["e_m"]) - e_min) / res - x0
        rn = (n_max - float(pose["n_m"])) / res - y0
        yy, xx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float32)
        dist = (xx - cn) ** 2 + (yy - rn) ** 2
        b = best[y0:y1, x0:x1]
        take = mask & (dist < b)
        if take.any():
            canvas[y0:y1, x0:x1][take] = warped[take]
            b[take] = dist[take]
            used += 1
    # OpenCV places pixel centres on integer coordinates: cell (0, 0) is centred on (n_max, e_min)
    if not used:
        return None
    return GroundImage(canvas, np.isfinite(best), n_max + res / 2, e_min - res / 2, res, used)
