"""Per-box ranging for the end-to-end pipeline.

Each box's bottom-centre (the object's ground contact) is put on flat ground along its viewing
ray (`evaluate.auair_world.ground_offset_oblique`). Range is the slant distance from the camera;
azimuth is the bearing of the ground point from the drone, clockwise from north; relative
azimuth is that bearing minus the drone's heading (-180 to 180); depression is the line of
sight's angle below the horizon. Range sigma is empirical: a relative spread per depression
band times range (configs/run.yaml, D-517). Flag: invalid (no ground point), uncertain
(sigma / range above the threshold), reliable otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerialrange.evaluate.auair_world import ground_offset_oblique

FIELDS = (
    "ground_n_m",
    "ground_e_m",
    "range_m",
    "range_sigma_m",
    "azimuth_deg",
    "relative_azimuth_deg",
    "depression_deg",
)


@dataclass(frozen=True)
class CameraPose:
    """Camera position (local north / east, metres), true height above the ground (m), heading
    (radians clockwise from north) and tilt below the horizon (degrees; NaN if unknown)."""

    n_m: float
    e_m: float
    h_m: float
    yaw_rad: float
    tilt_deg: float


@dataclass(frozen=True)
class RangingModel:
    focal_px: float
    cx: float
    cy: float
    edges_deg: NDArray
    rel_sigma: NDArray
    reliable_max_rel: float

    def band(self, depression_deg: NDArray) -> NDArray:
        """Index of the depression band of each angle."""
        return np.clip(
            np.searchsorted(self.edges_deg[1:], depression_deg, side="right"),
            0,
            len(self.rel_sigma) - 1,
        )


def range_boxes(boxes_ltwh: NDArray, pose: CameraPose, model: RangingModel) -> dict[str, NDArray]:
    """Ranging fields for boxes (n, 4) in one frame."""
    b = np.asarray(boxes_ltwh, float).reshape(-1, 4)
    n = len(b)
    u, v = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3]
    out: dict[str, NDArray] = {k: np.full(n, np.nan) for k in FIELDS}
    out["valid"] = np.zeros(n, bool)
    first = "no_tilt" if not np.isfinite(pose.tilt_deg) else "above_horizon"
    out["reason"] = np.full(n, first, object)
    out["flag"] = np.full(n, "invalid", object)
    if not n or not np.isfinite(pose.tilt_deg):
        return out
    dn, de = ground_offset_oblique(
        u, v, pose.h_m, model.focal_px, pose.yaw_rad, pose.tilt_deg, model.cx, model.cy
    )
    ok = np.isfinite(dn) & np.isfinite(de)
    horiz = np.hypot(dn, de)
    rng = np.hypot(horiz, pose.h_m)
    dep = np.degrees(np.arctan2(pose.h_m, horiz))
    az = np.degrees(np.arctan2(de, dn)) % 360.0
    rel_az = (az - np.degrees(pose.yaw_rad) + 180.0) % 360.0 - 180.0
    sig = model.rel_sigma[model.band(np.where(ok, dep, 0.0))] * rng
    out.update(
        ground_n_m=np.where(ok, pose.n_m + dn, np.nan),
        ground_e_m=np.where(ok, pose.e_m + de, np.nan),
        range_m=np.where(ok, rng, np.nan),
        range_sigma_m=np.where(ok, sig, np.nan),
        azimuth_deg=np.where(ok, az, np.nan),
        relative_azimuth_deg=np.where(ok, rel_az, np.nan),
        depression_deg=np.where(ok, dep, np.nan),
    )
    out["valid"] = ok
    out["reason"] = np.where(ok, "ok", out["reason"]).astype(object)
    with np.errstate(invalid="ignore"):
        good = sig / rng <= model.reliable_max_rel
    out["flag"] = np.where(ok, np.where(good, "reliable", "uncertain"), "invalid").astype(object)
    return out
