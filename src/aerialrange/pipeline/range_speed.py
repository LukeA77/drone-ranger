"""Ranging and speed for tracked boxes, frame by frame: the stage after the tracker.

`RangeSpeedStage.step` takes one frame's tracked boxes (track id, group, box), the camera pose
and the frame time; it places each box on the ground (`pipeline/ranging.py`), feeds the ground
points to one motion filter per track (`pipeline/filterbank.py`) and returns the rows of
tracks.csv for that frame, with the display rule's verdict on each speed. The live pipeline
(`pipeline/run.py`) and `replay` (the same stage over a saved tracks.csv, so ranging and filter
settings can be compared on identical tracks) both use it.

Measurement errors for the filters (`kalman.meas_model`):
- `relative` (version 1, D-518): isotropic, sigma = `meas_rel_sigma` (by depression band) x
  slant range.
- `propagated` (D-735): the full 2 x 2 covariance of the ground point from the errors of its
  inputs (`geometry/ground_cov.py`): the contact pixel (sigma_u = kappa_u x box width,
  sigma_v = kappa_v x box height, per group, from the detector's measured box noise), the
  camera tilt, height and heading (`meas_sigma`; a per-frame `tilt_sigma_deg` in the flight
  data replaces the configured tilt sigma). Each kappa is a number per group or, since the
  relative box noise is larger on small boxes, a table by box height per group:
  `{edges_px: [0, 16, 32, 64, 128, .inf], values: [...], overall: ...}` (an empty bin, NaN,
  takes `overall`).

Footprint centre (`footprint`, D-736): when configured, each box's ground point is moved from
the box's bottom edge to the footprint centre along the camera heading
(`pipeline/footprint.py`), using the speed filter's heading for that track from the previous
frame (uniform when there is none); range, bearing and depression follow the moved point, and
the offset's variance is added to the range sigma. The speed filter keeps the bottom-edge point:
moving it as well made slow vehicles' speeds worse, because their heading swings and the offset
with it (D-736). Optionally (`footprint.contact`, per group: `shift_px`, `raise_m`, D-743) the
bottom-edge point is first placed under the footprint's near edge, since a box's lowest visible
point sits a little above it (a car's body ends above the ground).

Calibrated intervals (`intervals`, D-737): when configured, every row also carries the
half-widths of the range and speed intervals at 68, 90 and 95 % (`pipeline/intervals.py`), and
the display rule judges a speed by its calibrated 68 % half-width instead of the filter's sigma.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerialrange.evaluate.auair_world import ground_offset_oblique
from aerialrange.geometry.ground_cov import ground_covariance
from aerialrange.pipeline.filterbank import make_filter_bank
from aerialrange.pipeline.footprint import FootprintSize, forward_offset
from aerialrange.pipeline.intervals import LEVEL_KEYS, IntervalCalibration
from aerialrange.pipeline.ranging import CameraPose, RangingModel, range_boxes
from aerialrange.pipeline.speed_display import SpeedDisplayRule

Tracked = tuple[int, str, float, float, float, float]  # id, group, left, top, width, height
NO_SPEED = (np.nan, np.nan, np.nan, np.nan, 0, np.nan, 0)
DEFAULT_DT_S = 0.2  # first frame or unknown time: AU-AIR's frame interval (D-613)


def ranging_model(get: Callable[[str], Any]) -> RangingModel:
    cam, rs = get("camera"), get("range_sigma_rel_by_depression")
    return RangingModel(
        float(cam["focal_px"]),
        float(cam["cx"]),
        float(cam["cy"]),
        np.array(rs["edges_deg"], float),
        np.array(rs["rel"], float),
        float(get("reliable_max_rel_sigma")),
    )


def display_rule(sd: dict) -> SpeedDisplayRule | None:
    if sd.get("sigma_max_ms") is None or sd.get("min_age") is None:
        return None
    return SpeedDisplayRule(
        float(sd["sigma_max_ms"]),
        int(sd["min_age"]),
        int(sd["nis_window"]),
        float(sd["nis_conf"]),
        str(sd.get("nis_test", "two_sided")),
    )


def display_rules(sd: dict) -> dict[str, SpeedDisplayRule | None] | SpeedDisplayRule | None:
    """One rule for every group, or, with `by_group` (D-739), one per group: a group not
    listed, or listed with a null sigma, shows no speed."""
    if "by_group" not in sd:
        return display_rule(sd)
    shared = {k: v for k, v in sd.items() if k != "by_group"}
    return {g: display_rule({**shared, **(v or {})}) for g, v in sd["by_group"].items()}


def exact_pose_overrides(kalman: dict) -> dict:
    """Config overrides for a flight whose camera poses are exact (synthetic scenes): the pose
    error terms of propagated measurement errors set to zero; nothing for other models."""
    if kalman.get("meas_model") != "propagated":
        return {}
    exact = {**kalman["meas_sigma"], "tilt_deg": 0.0, "h_m": 0.0, "yaw_deg": 0.0}
    return {"kalman": {"meas_sigma": exact}}


def kappa_at(spec: float | dict, height_px: float) -> float:
    """Relative box noise for a box of this height: a number, or a table by box height."""
    if not isinstance(spec, dict):
        return float(spec)
    edges = np.asarray(spec["edges_px"], float)
    b = int(np.clip(np.searchsorted(edges, height_px, side="right") - 1, 0, len(edges) - 2))
    v = float(spec["values"][b])
    return v if np.isfinite(v) else float(spec["overall"])


class RangeSpeedStage:
    def __init__(
        self,
        model: RangingModel,
        kalman: dict,
        speed_display: dict,
        footprint: dict | None = None,
        intervals: dict | None = None,
    ) -> None:
        self.model = model
        self.intervals = IntervalCalibration.from_mapping(intervals) if intervals else None
        on = bool(footprint and footprint.get("enabled"))
        self.sizes = (
            {g: FootprintSize.from_mapping(v) for g, v in footprint["sizes"].items()}
            if on
            else None
        )
        self.contact: dict[str, tuple[float, float]] = (
            {
                g: (float(v["shift_px"]), float(v["raise_m"]))
                for g, v in (footprint.get("contact") or {}).items()
            }
            if on
            else {}
        )
        self.heading: dict[int, tuple[float, float]] = {}  # id -> filter heading, sigma (deg)
        self.meas_model = kalman.get("meas_model", "relative")
        if self.meas_model == "relative":
            self.meas_rel = np.array(
                kalman["meas_rel"] if "meas_rel" in kalman else kalman["meas_rel_sigma"], float
            )
        elif self.meas_model == "propagated":
            ms = kalman["meas_sigma"]
            self.kappa = {g: (ms["kappa_u"][g], ms["kappa_v"][g]) for g in ms["kappa_u"]}
            self.sig_tilt, self.sig_h, self.sig_yaw = (
                float(ms["tilt_deg"]),
                float(ms["h_m"]),
                float(ms["yaw_deg"]),
            )
        else:
            raise ValueError(f"unknown meas_model {self.meas_model!r} (relative or propagated)")
        self.bank = make_filter_bank(kalman, int(speed_display["nis_window"]))
        self.rule = display_rules(speed_display)
        self.t_prev: float | None = None

    @classmethod
    def from_config(cls, get: Callable[[str], Any]) -> RangeSpeedStage:
        def optional(key: str) -> Any:
            try:
                return get(key)
            except KeyError:  # configurations written before D-736 / D-737
                return None

        return cls(
            ranging_model(get),
            get("kalman"),
            get("speed_display"),
            optional("footprint"),
            optional("intervals"),
        )

    def contact_shift(
        self, rg: dict[str, NDArray], boxes: NDArray, groups: Sequence[str], pose: CameraPose
    ) -> None:
        """Place the bottom-edge ground point of groups with `footprint.contact` under the
        footprint's near edge (in place, D-743): the box's lowest visible point lies a few
        pixels and some centimetres above that edge, so the ray through the bottom-centre moved
        down by `shift_px` is intersected with a plane `raise_m` above the ground and the point
        dropped to the ground."""
        if not self.contact:
            return
        m = self.model
        for j in np.flatnonzero(rg["valid"]):
            c = self.contact.get(groups[j])
            if c is None or c[1] >= pose.h_m:
                continue
            u = boxes[j, 0] + boxes[j, 2] / 2
            v = boxes[j, 1] + boxes[j, 3] + c[0]
            dn, de = ground_offset_oblique(
                np.array([u]),
                np.array([v]),
                pose.h_m - c[1],
                m.focal_px,
                pose.yaw_rad,
                pose.tilt_deg,
                m.cx,
                m.cy,
            )
            if np.isfinite(dn[0]) and np.isfinite(de[0]):
                rg["ground_n_m"][j] = pose.n_m + dn[0]
                rg["ground_e_m"][j] = pose.e_m + de[0]

    def footprint_shift(
        self,
        rg: dict[str, NDArray],
        ids: NDArray,
        groups: Sequence[str],
        pose: CameraPose,
        boxes: NDArray | None = None,
    ) -> NDArray:
        """Move valid ground points to the footprint centre (in place); returns each box's
        offset variance (m^2, 0 where not moved)."""
        var = np.zeros(len(ids))
        if self.sizes is None:
            return var
        if boxes is not None:
            self.contact_shift(rg, boxes, groups, pose)
        f = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
        for j in np.flatnonzero(rg["valid"]):
            size = self.sizes.get(groups[j])
            if size is None:
                continue
            hd, hs = self.heading.get(int(ids[j]), (np.nan, np.nan))
            rel = np.radians(hd) - pose.yaw_rad
            m, v = forward_offset(size, [rel], [np.radians(hs)])
            rg["ground_n_m"][j] += m[0] * f[0]
            rg["ground_e_m"][j] += m[0] * f[1]
            var[j] = v[0]
        ok = rg["valid"]
        dn, de = rg["ground_n_m"] - pose.n_m, rg["ground_e_m"] - pose.e_m
        horiz = np.hypot(dn, de)
        rng = np.hypot(horiz, pose.h_m)
        az = np.degrees(np.arctan2(de, dn)) % 360.0
        sig = np.sqrt(rg["range_sigma_m"] ** 2 + var)
        with np.errstate(invalid="ignore"):
            good = sig / rng <= self.model.reliable_max_rel
        rg.update(
            range_m=np.where(ok, rng, np.nan),
            range_sigma_m=np.where(ok, sig, np.nan),
            azimuth_deg=np.where(ok, az, np.nan),
            relative_azimuth_deg=np.where(
                ok, (az - np.degrees(pose.yaw_rad) + 180.0) % 360.0 - 180.0, np.nan
            ),
            depression_deg=np.where(ok, np.degrees(np.arctan2(pose.h_m, horiz)), np.nan),
        )
        rg["flag"] = np.where(ok, np.where(good, "reliable", "uncertain"), "invalid").astype(object)
        return var

    def measurement_errors(
        self,
        boxes: NDArray,
        groups: Sequence[str],
        rg: dict[str, NDArray],
        pose: CameraPose,
        tilt_sigma_deg: float,
    ) -> NDArray:
        """Per-box measurement error for the filters: sigmas (n,) or covariances (n, 2, 2)."""
        if self.meas_model == "relative":
            band = self.model.band(np.nan_to_num(rg["depression_deg"]))
            return self.meas_rel[band] * np.nan_to_num(rg["range_m"])
        hgt = boxes[:, 3]
        ku = np.array([kappa_at(self.kappa[g][0], h) for g, h in zip(groups, hgt, strict=True)])
        kv = np.array([kappa_at(self.kappa[g][1], h) for g, h in zip(groups, hgt, strict=True)])
        st = tilt_sigma_deg if np.isfinite(tilt_sigma_deg) else self.sig_tilt
        m = self.model
        _, cov = ground_covariance(
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3],
            pose.h_m,
            m.focal_px,
            pose.yaw_rad,
            pose.tilt_deg,
            m.cx,
            m.cy,
            ku * boxes[:, 2],
            kv * boxes[:, 3],
            st,
            self.sig_h,
            self.sig_yaw,
        )
        return cov.reshape(-1, 2, 2)

    def step(
        self,
        k: int,
        t_s: float,
        pose: CameraPose,
        tracked: Sequence[Tracked],
        tilt_sigma_deg: float = np.nan,
    ) -> list[dict]:
        boxes = np.array([t[2:] for t in tracked], float).reshape(-1, 4)
        rg = range_boxes(boxes, pose, self.model)
        dt = (t_s - self.t_prev) if self.t_prev is not None and np.isfinite(t_s) else DEFAULT_DT_S
        self.t_prev = t_s
        valid = rg["valid"]
        groups = [t[1] for t in tracked]
        ids = np.array([t[0] for t in tracked], int)
        msig = self.measurement_errors(boxes, groups, rg, pose, tilt_sigma_deg)
        # the speed filter takes the box's bottom-edge point; only the reported range moves to
        # the footprint centre (D-736: a slow object's heading swings, so a heading-dependent
        # offset would add apparent motion)
        z = np.column_stack([rg["ground_n_m"], rg["ground_e_m"]])
        self.footprint_shift(rg, ids, groups, pose, boxes)
        fb = self.bank.step(
            max(dt, 1e-3),
            ids[valid],
            z[valid],
            msig[valid],
            [t[1] for t, ok in zip(tracked, valid, strict=True) if ok],
        )
        speed = {
            int(i): (
                fb["speed_ms"][j],
                fb["speed_sigma_ms"][j],
                fb["heading_deg"][j],
                fb["heading_sigma_deg"][j],
                int(fb["age"][j]),
                fb["nis_mean"][j],
                int(fb["nis_n"][j]),
            )
            for j, i in enumerate(fb["id"])
        }
        # the filter's heading per live track, for the next frame's footprint correction
        self.heading = {i: (v[2], v[3]) for i, v in speed.items()}
        rows = []
        for j, (tid, group, x, y, w, h) in enumerate(tracked):
            sp, ss, hd, hs, age, nis_m, nis_n = speed.get(tid, NO_SPEED)
            rh = sh = dict.fromkeys(LEVEL_KEYS, np.nan)
            if self.intervals is not None and valid[j]:
                dep, rng = rg["depression_deg"][j], rg["range_m"][j]
                rh = self.intervals.half_widths("range", group, dep, rng, rg["range_sigma_m"][j])
                sh = self.intervals.half_widths("speed", group, dep, rng, ss)
            judged = sh["0.68"] if self.intervals is not None else ss
            rule = self.rule.get(group) if isinstance(self.rule, dict) else self.rule
            shown = bool(rule is not None and valid[j] and rule.ok(judged, age, nis_m, nis_n))
            rows.append(
                {
                    "frame": k,
                    "t_s": t_s,
                    "track_id": tid,
                    "class": group,
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "range_m": rg["range_m"][j],
                    "range_sigma_m": rg["range_sigma_m"][j],
                    "azimuth_deg": rg["azimuth_deg"][j],
                    "relative_azimuth_deg": rg["relative_azimuth_deg"][j],
                    "depression_deg": rg["depression_deg"][j],
                    "speed_ms": sp,
                    "speed_sigma_ms": ss,
                    "valid": bool(valid[j]),
                    "reason": rg["reason"][j],
                    "flag": rg["flag"][j],
                    "heading_deg": hd,
                    "heading_sigma_deg": hs,
                    "track_age": age,
                    "nis_mean": nis_m,
                    "speed_shown": shown,
                    **{f"range_half_{lv[2:]}_m": rh[lv] for lv in LEVEL_KEYS},
                    **{f"speed_half_{lv[2:]}_ms": sh[lv] for lv in LEVEL_KEYS},
                }
            )
        return rows


def pose_of(meta: pd.DataFrame, k: int) -> tuple[CameraPose, float, float]:
    """Camera pose, time and tilt sigma (NaN unless the flight data has `tilt_sigma_deg`) of
    frame `k` from flight data indexed by frame (NaN if the frame is absent)."""
    if k not in meta.index:
        return CameraPose(np.nan, np.nan, np.nan, 0.0, np.nan), np.nan, np.nan
    m = meta.loc[k]
    pose = CameraPose(*(float(m[c]) for c in ("n_m", "e_m", "h_m", "yaw_rad", "tilt_deg")))
    ts = float(m["tilt_sigma_deg"]) if "tilt_sigma_deg" in m.index else np.nan
    return pose, float(m["t_s"]), ts


def replay(tracks: pd.DataFrame, meta: pd.DataFrame, stage: RangeSpeedStage) -> pd.DataFrame:
    """Run the stage over a saved tracks.csv: every frame of `meta` (indexed by frame) in order,
    with that frame's tracked boxes in their saved order."""
    by_f = {int(f): g for f, g in tracks.groupby("frame", sort=False)}
    rows: list[dict] = []
    for k in sorted(int(x) for x in meta.index):
        g = by_f.get(k)
        tracked: list[Tracked] = (
            []
            if g is None
            else [
                (int(r.track_id), str(r["class"]), r.left, r.top, r.width, r.height)
                for _, r in g.iterrows()
            ]
        )
        pose, t_s, ts = pose_of(meta, k)
        rows += stage.step(k, t_s, pose, tracked, ts)
    return pd.DataFrame(rows)
