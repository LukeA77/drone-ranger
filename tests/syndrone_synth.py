"""Synthetic SynDrone-like scenes for tests (labelled synthetic; no dataset content).

Ray-casts a flat road plane and axis-aligned or yawed boxes into planar depth and semantic
images for a CARLA camera, using the same conventions as the loader.
"""

from __future__ import annotations

import numpy as np

from aerialrange.data.syndrone import CameraRecord
from aerialrange.geometry.carla import carla_camera, ue_from_ned


def box_corners(centre, half, yaw_deg=0.0):
    """SynDrone corner order (get_corners_from_bb) for a box yawed about z."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    signs = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]
    return np.array([np.asarray(centre) + R @ (np.array(sg) * half) for sg in signs])


def render(rec: CameraRecord, z0: float, ground_z: float, boxes, w=192, h=108, road_tag=7):
    """boxes: list of (corners (8, 3), tag). Returns camera, depth (planar), semantic."""
    cam = carla_camera(w, h, 90.0)
    pose = rec.pose(z0)
    R = pose.R_wb @ cam.R_bc
    C = ue_from_ned(pose.p_w)
    v, u = np.mgrid[0:h, 0:w].astype(float)
    dc = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1)
    d = ue_from_ned(dc.reshape(-1, 3) @ R.T)  # planar parameter: camera z of d is 1
    depth = np.full(len(d), 1000.0)
    sem = np.full(len(d), 13, np.uint8)  # sky
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (ground_z - C[2]) / d[:, 2]
    ok = tg > 0
    depth[ok], sem[ok] = tg[ok], road_tag
    for corners, tag in boxes:
        cen = corners.mean(0)
        ax = np.stack([corners[0] - corners[4], corners[0] - corners[2], corners[0] - corners[1]])
        half = np.linalg.norm(ax, axis=1) / 2
        ax = ax / (2 * half[:, None])
        o, dl = (C - cen) @ ax.T, d @ ax.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / dl, (half - o) / dl
        tmin = np.nanmax(np.minimum(t1, t2), 1)
        tmax = np.nanmin(np.maximum(t1, t2), 1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
        depth[hit], sem[hit] = tmin[hit], tag
    return cam, depth.reshape(h, w).astype(np.float32), sem.reshape(h, w)
