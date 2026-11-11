"""The simulated drones' camera (Phase 14, D-752): footprint, passes and detections.

Everything here comes from measurement (results/multi_drone/sensor_<run>.json, written by
scripts/multidrone_sensor_fit.py from the Phase 10 SynDrone test outputs): per camera height,
the probability that a person under the camera is detected during one pass, false alarms per
km2 of ground passed over, and the size of a detection's position error. The camera geometry is
SynDrone's (1920 x 1080, 90 degree horizontal field of view, the height-to-tilt pairing of its
three cameras), so the simulated footprint is the one the measurement was made with. Only the
ground at least the lowest reliable depression below the horizon counts as searched.

Detections are produced in the ROS 2 node's output format (drone_ranger_msgs/TrackedObjectArray,
as dictionaries with the same field names), and the search map reads them back from that
format (`observation_detections`), so the simulation consumes what the real node publishes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerialrange.pipeline.mosaic import Camera, frame_quad
from aerialrange.search.belief import Detection

SYNDRONE_CAMERA = Camera(focal_px=960.0, cx=960.0, cy=540.0, width_px=1920, height_px=1080)
TILT_OF_HEIGHT = {20: 30.0, 50: 60.0, 80: 90.0}  # SynDrone camera files (data/syndrone.py)


@dataclass(frozen=True)
class SensorModel:
    height_m: int
    tilt_deg: float
    pd_per_pass: float
    fa_per_m2: float
    pos_sigma_m: float
    min_dep_deg: float
    camera: Camera = SYNDRONE_CAMERA
    # Detection probability by pass length: a pass of n frames falls in the first bin whose
    # upper edge is at least n (longer passes than the last edge use the last bin). Empty:
    # pd_per_pass for every pass.
    pd_edges_frames: tuple[float, ...] = ()
    pd_bins: tuple[float, ...] = ()

    @classmethod
    def from_results(cls, path: Path, height_m: int) -> SensorModel:
        r = json.loads(Path(path).read_text())
        h = r["by_height"][f"h{height_m}"]
        bins = sorted(
            ((float(k.split("-")[1]), float(v["pd"])) for k, v in h["pd_by_pass_frames"].items()),
        )
        sig = h["range_half_68_m_median"]
        return cls(
            height_m=height_m,
            tilt_deg=TILT_OF_HEIGHT[height_m],
            pd_per_pass=float(h["pd_per_pass"]),
            fa_per_m2=float(h["false_alarms_per_km2"]) / 1e6,
            # no matched person at this height: no true detection is ever placed, so the
            # position error only shapes false alarms' likelihood; NaN marks it unmeasured
            pos_sigma_m=float(sig) if sig is not None else float("nan"),
            min_dep_deg=float(r["min_depression_deg"]),
            pd_edges_frames=tuple(b[0] for b in bins),
            pd_bins=tuple(b[1] for b in bins),
        )

    @property
    def can_detect(self) -> bool:
        return self.pd_per_pass > 0 and np.isfinite(self.pos_sigma_m)

    def pd_for(self, frames: NDArray) -> NDArray:
        """Detection probability of passes lasting `frames` camera frames."""
        f = np.asarray(frames, float)
        if not self.pd_bins:
            return np.full(f.shape, self.pd_per_pass)
        k = np.searchsorted(np.asarray(self.pd_edges_frames), f, side="left")
        return np.asarray(self.pd_bins)[np.minimum(k, len(self.pd_bins) - 1)]

    def footprint(self, n_m: float, e_m: float, yaw_rad: float) -> NDArray:
        """Usable ground footprint (4 x 2, north / east) of the camera at this pose; the camera
        looks along the flight heading `yaw_rad` (from north, clockwise). At a fixed height and
        tilt the footprint only turns with the heading, so the one at heading 0 is rotated
        (equal to `exact_footprint`, tested)."""
        local = _local_footprint(self)
        if len(local) == 0:
            return local
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        n = local[:, 0] * c - local[:, 1] * s
        e = local[:, 0] * s + local[:, 1] * c
        return np.c_[n + n_m, e + e_m]

    def exact_footprint(self, n_m: float, e_m: float, yaw_rad: float) -> NDArray:
        """The footprint from the pipeline's ray model at this pose (mosaic.frame_quad)."""
        pose = pd.Series(
            {
                "n_m": n_m,
                "e_m": e_m,
                "h_m": float(self.height_m),
                "yaw_rad": yaw_rad,
                "tilt_deg": self.tilt_deg,
            }
        )
        _, ne = frame_quad(pose, self.camera, self.min_dep_deg)
        return ne


_LOCAL: dict[tuple, NDArray] = {}


def _local_footprint(s: SensorModel) -> NDArray:
    key = (s.height_m, s.tilt_deg, s.min_dep_deg, s.camera)
    if key not in _LOCAL:
        _LOCAL[key] = s.exact_footprint(0.0, 0.0, 0.0)
    return _LOCAL[key]


def tracked_object_array(
    stamp_s: float,
    frame_index: int,
    dets: list[tuple[float, float]],
    sigma_m: float,
    first_track_id: int,
) -> dict:
    """Detections as a TrackedObjectArray message (field names as drone_ranger_msgs). Positions
    are ENU (x east, y north); covariance is 2 x 2 (east, north), row-major. Only the fields the
    search map reads carry simulated values; the rest are set as the node sets them for a
    reliable person with no displayed speed."""
    objs = []
    for k, (n, e) in enumerate(dets):
        objs.append(
            {
                "track_id": first_track_id + k,
                "object_class": "person",
                "bbox": None,
                "valid": True,
                "reason": "ok",
                "reliability": "reliable",
                "position": {"x": e, "y": n, "z": 0.0},
                "position_covariance": [sigma_m**2, 0.0, 0.0, sigma_m**2],
                "range_m": float("nan"),
                "range_sigma_m": float("nan"),
                "range_half_width_m": [sigma_m, float("nan"), float("nan")],
                "velocity_valid": False,
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "velocity_covariance": [0.0, 0.0, 0.0, 0.0],
            }
        )
    return {
        "header": {"stamp": stamp_s, "frame_id": "enu"},
        "frame_index": frame_index,
        "objects": objs,
    }


def observation_detections(msg: dict) -> tuple[Detection, ...]:
    """Person detections the search map uses, read from a TrackedObjectArray: valid, reliable
    people, at their ground position with the calibrated 68 % half-width as the position error
    (the map's ellipse, D-750), or the raw covariance when no calibration is present."""
    out = []
    for o in msg["objects"]:
        if o["object_class"] != "person" or not o["valid"] or o["reliability"] != "reliable":
            continue
        e, n = float(o["position"]["x"]), float(o["position"]["y"])
        half = float(o["range_half_width_m"][0])
        c = o["position_covariance"]
        if np.isfinite(half) and half > 0:
            cov = (half**2, 0.0, 0.0, half**2)
        else:
            cov = (float(c[3]), float(c[2]), float(c[1]), float(c[0]))  # (n n, n e, e n, e e)
        out.append(Detection(n, e, cov))
    return tuple(out)
