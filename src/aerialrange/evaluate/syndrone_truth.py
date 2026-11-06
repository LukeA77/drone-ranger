"""SynDrone ground truth per object and frame.

All positions are Unreal world (x, y, z up, metres) unless named `_ned`.

- `estimate_z0`: the per-frame base height that SynDrone adds to the camera height but does
  not save. Vehicles stand on the road, so the road just around each moving vehicle's
  footprint must lie at the height of the box's bottom face. Each vehicle gives one robust
  offset (median over its ring of road pixels, so a kerb or a mislabelled pixel cannot pull
  it); the offsets from all cameras of the frame are combined by a Huber M-estimate
  (scale from the median absolute deviation), which is the robust least-squares fit of one
  shared unknown. Its standard error and residuals are returned for the report.
- `object_truth`: per box, the instance mask (pixels of a compatible class whose world
  position from depth lies inside the box, with a margin), visibility (instance pixels over
  instance plus occluded pixels, among pixels whose ray passes through the box), the
  projected 2D box and the visible-mask 2D box, range to the footprint centre and to the
  footprint's near edge, depression angle and truncation.

Horizontal world positions from depth do not depend on `z0` (camera x, y and rotation are
known), so instance masks are built from them and are independent of the `z0` they check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aerialrange.data.syndrone import BOTTOM, RIDER, Boxes, CameraRecord
from aerialrange.geometry.camera import Camera
from aerialrange.geometry.carla import ned_from_ue, ue_from_ned

ROAD_TAGS = (6, 7)  # road line, road (labelmap.py); sidewalks excluded (kerb height)
TWO_WHEELERS = (104, 105)


@dataclass(frozen=True)
class TruthSettings:
    ring_m: float = 1.5  # road ring width around a vehicle footprint
    min_ring_px: int = 30  # road pixels needed for one vehicle's offset
    huber_k: float = 1.345  # Huber tuning constant (95 % efficiency under normal errors)
    inside_margin_m: float = 0.3  # margin when testing a depth point against a box
    occluder_margin_m: float = 0.3  # depth closer than the box entry by this = occluder
    min_instance_px: int = 4  # fewer instance pixels = not visible


@dataclass
class View:
    """One camera's data for one frame."""

    height: int
    record: CameraRecord
    depth: NDArray[np.float32]
    semantic: NDArray[np.uint8]
    camera: Camera
    _rel: NDArray[np.float32] | None = field(default=None, repr=False)

    def rel_points(self) -> NDArray[np.float32]:
        """World points minus the camera's Unreal position at z0 = 0, i.e. (x, y, z) with
        x, y absolute and z relative to the base height: world z = this z + z0."""
        if self._rel is None:
            self._rel = rel_world_points(self.camera, self.record, self.depth)
        return self._rel


def rel_world_points(cam: Camera, rec: CameraRecord, depth: NDArray) -> NDArray[np.float32]:
    """(H, W, 3) Unreal world points from planar depth, with z0 = 0."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    d = depth.astype(np.float32)
    xc = np.stack([(u - cam.cx) / cam.fx * d, (v - cam.cy) / cam.fy * d, d], axis=-1)
    pose = rec.pose(0.0)
    R = (pose.R_wb @ cam.R_bc).astype(np.float32)
    p_ned = xc @ R.T + pose.p_w.astype(np.float32)
    return ue_from_ned(p_ned).astype(np.float32)


def box_frames(corners: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Centres (N, 3), axes (N, 3, 3; rows x, y, z unit vectors) and half sizes (N, 3).

    From SynDrone's corner order: x = c0 - c4, y = c0 - c2, z = c0 - c1.
    """
    c = np.asarray(corners, float)
    vx, vy, vz = c[:, 0] - c[:, 4], c[:, 0] - c[:, 2], c[:, 0] - c[:, 1]
    half = np.stack([np.linalg.norm(v, axis=1) for v in (vx, vy, vz)], axis=1) / 2.0
    axes = np.stack([vx, vy, vz], axis=1) / np.maximum(2.0 * half[:, :, None], 1e-12)
    return c.mean(axis=1), axes, half


def huber_location(x: NDArray, k: float, iters: int = 50) -> tuple[float, float, float]:
    """Huber M-estimate of location with MAD scale. Returns (estimate, standard error, scale).

    Standard error from the M-estimator's asymptotic variance (Huber, 1981, Robust
    Statistics, Section 3.2): s^2 E[psi^2] / (n E[psi']^2).
    """
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - mu)))
    if n == 1 or s == 0.0:
        return mu, (0.0 if n > 1 else float("nan")), s
    for _ in range(iters):
        r = (x - mu) / s
        w = np.minimum(1.0, k / np.maximum(np.abs(r), 1e-12))
        new = float(np.sum(w * x) / np.sum(w))
        if abs(new - mu) < 1e-9 * max(1.0, abs(mu)):
            mu = new
            break
        mu = new
    r = (x - mu) / s
    psi = np.clip(r, -k, k)
    dpsi = (np.abs(r) <= k).astype(float)
    se = s * np.sqrt(np.mean(psi**2)) / max(np.mean(dpsi), 1e-12) / np.sqrt(n)
    return mu, float(se), s


@dataclass(frozen=True)
class Z0Fit:
    z0: float
    se: float
    scale: float
    n_boxes: int
    per_view: dict[int, tuple[float, int]]  # height -> (median offset, boxes)
    offsets: NDArray[np.float64]


def vehicle_offsets(view: View, boxes: Boxes, s: TruthSettings) -> NDArray[np.float64]:
    """One z0 estimate per vehicle actor seen with enough road around it.

    Road points are indexed by horizontal position (a k-d tree), which does not depend on
    z0, so each vehicle only looks at the road near it.
    """
    P = view.rel_points()
    road = np.isin(view.semantic, ROAD_TAGS) & (view.depth < 999.0)
    pts = P[road]
    sel = (~boxes.level) & np.isin(boxes.tags, (100, 101, 102))
    if len(pts) == 0 or not sel.any():
        return np.zeros(0)
    tree = cKDTree(pts[:, :2])
    cen, axes, half = box_frames(boxes.corners[sel])
    bottoms = boxes.corners[sel][:, BOTTOM, 2].mean(axis=1)
    out = []
    for c, a, hh, zb in zip(cen, axes, half, bottoms, strict=True):
        radius = float(np.hypot(hh[0] + s.ring_m, hh[1] + s.ring_m))
        idx = tree.query_ball_point(c[:2], radius)
        if len(idx) < s.min_ring_px:
            continue
        near = pts[idx]
        ring = _inside_footprint(near[:, :2], c, a, hh, s.ring_m)
        ring &= ~_inside_footprint(near[:, :2], c, a, hh, 0.0)
        if ring.sum() < s.min_ring_px:
            continue
        out.append(zb - float(np.median(near[ring, 2])))
    return np.asarray(out, float)


def estimate_z0(views: list[View], boxes: Boxes, s: TruthSettings | None = None) -> Z0Fit:
    s = s or TruthSettings()
    per_view: dict[int, tuple[float, int]] = {}
    allo = []
    for v in views:
        o = vehicle_offsets(v, boxes, s)
        per_view[v.height] = (float(np.median(o)) if len(o) else float("nan"), len(o))
        allo.append(o)
    offs = np.concatenate(allo) if allo else np.zeros(0)
    mu, se, sc = huber_location(offs, s.huber_k)
    return Z0Fit(mu, se, sc, len(offs), per_view, offs)


def project_ue(cam: Camera, rec: CameraRecord, z0: float, X_ue: NDArray) -> tuple[NDArray, NDArray]:
    """Pixels (N, 2) and planar depth (N,) of Unreal world points."""
    pose = rec.pose(z0)
    R = pose.R_wb @ cam.R_bc
    Xc = (ned_from_ue(X_ue) - pose.p_w) @ R
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([cam.fx * Xc[:, 0] / z + cam.cx, cam.fy * Xc[:, 1] / z + cam.cy], axis=1)
    return uv, z


def _inside_footprint(xy: NDArray, c: NDArray, a: NDArray, hh: NDArray, m: float) -> NDArray:
    """Points whose horizontal position lies within the box footprint plus margin `m`."""
    d = xy - c[:2]
    ax = a[0, :2] / max(np.linalg.norm(a[0, :2]), 1e-9)
    ay = a[1, :2] / max(np.linalg.norm(a[1, :2]), 1e-9)
    return (np.abs(d @ ax) <= hh[0] + m) & (np.abs(d @ ay) <= hh[1] + m)


def compatible_tags(tag: int) -> tuple[int, ...]:
    return (tag, RIDER) if tag in TWO_WHEELERS else (tag,)


def object_truth(view: View, boxes: Boxes, z0: float, s: TruthSettings | None = None) -> list[dict]:
    """Truth rows for every box at least partly in front of and inside the image."""
    s = s or TruthSettings()
    cam, rec = view.camera, view.record
    H, W = view.depth.shape
    P = view.rel_points()
    pose = rec.pose(z0)
    C_ue = ue_from_ned(pose.p_w)
    R = pose.R_wb @ cam.R_bc
    cen, axes, half = box_frames(boxes.corners)
    rows = []
    for i in range(len(boxes)):
        uv, zc = project_ue(cam, rec, z0, boxes.corners[i])
        if np.any(zc <= 0.1):
            continue  # behind or straddling the camera plane: not usable
        u0, v0 = np.floor(uv.min(0)).astype(int)
        u1, v1 = np.ceil(uv.max(0)).astype(int)
        cu0, cv0, cu1, cv1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
        if cu1 <= cu0 or cv1 <= cv0:
            continue
        area = float((u1 - u0) * (v1 - v0))
        trunc = 1.0 - (cu1 - cu0) * (cv1 - cv0) / max(area, 1e-9)
        # Box-local coordinates of the depth points in the projected rectangle.
        # Instance test on horizontal position only (independent of z0, see module notes).
        Pw = P[cv0:cv1, cu0:cu1].reshape(-1, 3).astype(float)
        inside = _inside_footprint(Pw[:, :2], cen[i], axes[i], half[i], s.inside_margin_m)
        tags = view.semantic[cv0:cv1, cu0:cu1].reshape(-1)
        inst = inside & np.isin(tags, compatible_tags(int(boxes.tags[i])))
        # Rays through the box: slab test in box-local coordinates, planar parameter t.
        vv, uu = np.mgrid[cv0:cv1, cu0:cu1]
        dc = np.stack(
            [(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones_like(uu, float)], axis=-1
        ).reshape(-1, 3)
        d_ue = ue_from_ned(dc @ R.T)
        o = (C_ue - cen[i]) @ axes[i].T
        dl = d_ue @ axes[i].T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half[i] - o) / dl
            t2 = (half[i] - o) / dl
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hits = (tmax >= tmin) & (tmax > 0)
        dep = view.depth[cv0:cv1, cu0:cu1].reshape(-1)
        occ = hits & ~inst & (dep < tmin - s.occluder_margin_m)
        n_inst, n_occ = int(inst.sum()), int(occ.sum())
        vis = n_inst / (n_inst + n_occ) if (n_inst + n_occ) else 0.0
        if n_inst >= s.min_instance_px:
            iv, iu = np.nonzero(inst.reshape(cv1 - cv0, cu1 - cu0))
            mbox = [cu0 + iu.min(), cv0 + iv.min(), cu0 + iu.max() + 1, cv0 + iv.max() + 1]
        else:
            mbox = [np.nan] * 4
        foot = boxes.corners[i, BOTTOM]
        fc = foot.mean(0)
        near = _nearest_on_polygon(foot[[0, 1, 3, 2], :2], C_ue[:2])
        near3 = np.array([near[0], near[1], fc[2]])
        rng = float(np.linalg.norm(fc - C_ue))
        horiz = float(np.linalg.norm(fc[:2] - C_ue[:2]))
        rows.append(
            {
                "id": int(boxes.ids[i]),
                "tag": int(boxes.tags[i]),
                "level": bool(boxes.level[i]),
                "n_instance_px": n_inst,
                "n_occluded_px": n_occ,
                "visibility": vis,
                "truncation": trunc,
                "box3d_proj": [float(u0), float(v0), float(u1), float(v1)],
                "box_mask": [float(x) for x in mbox],
                "range_m": rng,
                "range_near_m": float(np.linalg.norm(near3 - C_ue)),
                "depression_deg": float(np.degrees(np.arctan2(C_ue[2] - fc[2], horiz))),
                "foot_ue": fc.tolist(),
                "foot_ned": ned_from_ue(fc).tolist(),
            }
        )
    return rows


def _nearest_on_polygon(poly: NDArray, p: NDArray) -> NDArray:
    """Closest point to `p` on or inside a convex polygon (vertices in order)."""
    n = len(poly)
    crosses = []
    best, bd = poly[0], np.inf
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        e = b - a
        crosses.append(e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]))
        t = np.clip(np.dot(p - a, e) / max(np.dot(e, e), 1e-12), 0.0, 1.0)
        q = a + t * e
        dq = np.linalg.norm(p - q)
        if dq < bd:
            best, bd = q, dq
    c = np.asarray(crosses)
    inside = bool(np.all(c >= 0) or np.all(c <= 0))
    return np.asarray(p, float).copy() if inside else best
