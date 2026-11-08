"""Per-track ground-plane Kalman filters, stepped together (D-612).

Tracks come and go; each frame, every live filter is predicted by `dt` (constant velocity) and
those with a measurement are updated (`GroundKalmanBatch`). A new track is initialised by
two-point differencing on its second measurement. A filter is dropped after `max_unseen` frames
without a measurement (predict only in between).

Version 2 adds, per live filter: heading and its sigma, the number of
updates since initialisation (`age`), and the mean normalised innovation squared over the
last `nis_window` updates (`nis_mean`, `nis_n`), used to decide when a speed is shown.
Position and speed estimates are unchanged from version 1.

Measurement errors are given either as one standard deviation per measurement (k,), isotropic
as in version 1, or as a full covariance (k, 2, 2), e.g. propagated through the viewing
geometry (`geometry/ground_cov.py`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from aerialrange.geometry.imm import STATIONARY, IMMGround, IMMSettings
from aerialrange.geometry.kalman import GroundKalmanBatch


def as_covariances(sigma: NDArray, k: int) -> NDArray:
    """(k, 2, 2) measurement covariances from sigmas (k,) (isotropic) or covariances."""
    a = np.asarray(sigma, float)
    if a.ndim == 3:
        return a.reshape(k, 2, 2)
    return np.eye(2)[None] * (a.reshape(k) ** 2)[:, None, None]


EMPTY = {
    "id": np.zeros(0, int),
    "n_m": np.zeros(0),
    "e_m": np.zeros(0),
    "speed_ms": np.zeros(0),
    "speed_sigma_ms": np.zeros(0),
    "heading_deg": np.zeros(0),
    "heading_sigma_deg": np.zeros(0),
    "age": np.zeros(0, int),
    "nis_mean": np.zeros(0),
    "nis_n": np.zeros(0, int),
}


class TrackFilterBank:
    def __init__(self, q: float, max_unseen: int, nis_window: int = 10) -> None:
        self.q, self.max_unseen = q, max_unseen
        self.nis_window = nis_window
        self.age: dict[int, int] = {}
        self.nis: dict[int, deque[float]] = {}
        self.ids: list[int] = []
        self.bank: GroundKalmanBatch | None = None
        self.unseen: dict[int, int] = {}
        self.first: dict[int, tuple[NDArray, NDArray, float]] = {}  # id -> (z, R, age in s)

    def _add(self, i: int, z0: NDArray, r0: NDArray, z1: NDArray, r1: NDArray, dt: float) -> None:
        b = GroundKalmanBatch.from_two_points(z0[None], r0[None], z1[None], r1[None], dt, self.q)
        if self.bank is None or not self.ids:
            self.bank = b
        else:
            self.bank.x = np.vstack([self.bank.x, b.x])
            self.bank.P = np.concatenate([self.bank.P, b.P])
        self.ids.append(i)
        self.unseen[i] = 0
        self.age[i] = 0
        self.nis[i] = deque(maxlen=self.nis_window)

    def step(
        self,
        dt: float,
        ids: NDArray,
        z: NDArray,
        sigma: NDArray,
        groups: Sequence[str] | None = None,
    ) -> dict[str, NDArray]:
        """Advance by `dt` seconds with measurements z (k, 2) whose errors are `sigma` (k,)
        (metres, isotropic) or covariances (k, 2, 2) (m^2), for track ids (k,). Returns id,
        position, speed and speed sigma of every live filter. `groups` is not used (one
        motion model for every group)."""
        ids = np.asarray(ids, int)
        z = np.asarray(z, float).reshape(-1, 2)
        cov = as_covariances(sigma, len(z))
        meas = {int(i): (z[j], cov[j]) for j, i in enumerate(ids) if np.isfinite(z[j]).all()}
        if self.bank is not None and self.ids:
            self.bank.predict(dt)
            have = np.array([i in meas for i in self.ids])
            Z = np.array([meas[i][0] if i in meas else (0.0, 0.0) for i in self.ids])
            R = np.array([meas[i][1] if i in meas else np.eye(2) for i in self.ids])
            nis = self.bank.update(Z, R, have)
            for k, i in enumerate(self.ids):
                self.unseen[i] = 0 if i in meas else self.unseen[i] + 1
                if i in meas:
                    self.age[i] += 1
                    self.nis[i].append(float(nis[k]))
        self.first = {i: (v[0], v[1], v[2] + dt) for i, v in self.first.items()}
        for i, (zi, si) in meas.items():
            if i in self.unseen:
                continue
            if i in self.first:
                z0, s0, age = self.first.pop(i)
                self._add(i, z0, s0, zi, si, age)
            else:
                self.first[i] = (zi, si, 0.0)
        # drop filters unseen for too long, and first measurements never followed up
        keep = [k for k, i in enumerate(self.ids) if self.unseen[i] < self.max_unseen]
        if len(keep) < len(self.ids):
            for k, i in enumerate(self.ids):
                if k not in keep:
                    del self.unseen[i], self.age[i], self.nis[i]
            self.ids = [self.ids[k] for k in keep]
            self.bank.x, self.bank.P = self.bank.x[keep], self.bank.P[keep]
        self.first = {i: v for i, v in self.first.items() if v[2] < self.max_unseen * dt}
        if not self.ids:
            return {k: v.copy() for k, v in EMPTY.items()}
        sp, sp_sig = self.bank.speed()
        hd, hd_sig = self.bank.heading()
        return {
            "id": np.array(self.ids),
            "n_m": self.bank.x[:, 0].copy(),
            "e_m": self.bank.x[:, 1].copy(),
            "speed_ms": sp,
            "speed_sigma_ms": sp_sig,
            "heading_deg": hd,
            "heading_sigma_deg": hd_sig,
            "age": np.array([self.age[i] for i in self.ids], int),
            "nis_mean": np.array(
                [np.mean(self.nis[i]) if self.nis[i] else np.nan for i in self.ids]
            ),
            "nis_n": np.array([len(self.nis[i]) for i in self.ids], int),
        }


class IMMFilterBank:
    """Per-track IMM filters (`geometry/imm.py`): stationary, constant velocity and coordinated
    turn, with gating. Same interface and outputs as `TrackFilterBank`, plus `p_stationary`
    (the stationary model's probability). Initialisation and dropping as in version 1; a filter
    whose measurements fall outside its gate `reset_after` times in a row (an identity switch,
    seen from the filter) is restarted from two new measurements, as a new track. `settings`
    is one `IMMSettings` for every track or one per group (people and vehicles move
    differently); a track takes its group's settings when its filter starts."""

    def __init__(
        self,
        settings: IMMSettings | Mapping[str, IMMSettings],
        max_unseen: int,
        nis_window: int = 10,
    ) -> None:
        self.s, self.max_unseen, self.nis_window = settings, max_unseen, nis_window
        self.filters: dict[int, IMMGround] = {}
        self.unseen: dict[int, int] = {}
        self.age: dict[int, int] = {}
        self.nis: dict[int, deque[float]] = {}
        self.first: dict[int, tuple[NDArray, NDArray, float]] = {}  # id -> (z, R, age in s)
        self.resets = 0

    def _drop(self, i: int) -> None:
        for d in (self.filters, self.unseen, self.age, self.nis):
            d.pop(i, None)

    def settings_for(self, group: str | None) -> IMMSettings:
        if isinstance(self.s, IMMSettings):
            return self.s
        if group not in self.s:
            raise KeyError(f"no IMM settings for group {group!r}")
        return self.s[group]

    def step(
        self,
        dt: float,
        ids: NDArray,
        z: NDArray,
        sigma: NDArray,
        groups: Sequence[str] | None = None,
    ) -> dict[str, NDArray]:
        ids = np.asarray(ids, int)
        grp = {int(i): (groups[j] if groups is not None else None) for j, i in enumerate(ids)}
        z = np.asarray(z, float).reshape(-1, 2)
        cov = as_covariances(sigma, len(z))
        meas = {int(i): (z[j], cov[j]) for j, i in enumerate(ids) if np.isfinite(z[j]).all()}
        for i, f in list(self.filters.items()):
            f.predict(dt)
            if i in meas:
                zi, si = meas[i]
                if f.update(zi, si):
                    self.unseen[i] = 0
                    self.age[i] += 1
                    self.nis[i].append(f.nis)
                else:
                    self.unseen[i] += 1
                    if f.needs_reset:
                        self._drop(i)
                        self.resets += 1
                        self.first[i] = (zi, si, -dt)  # aged to 0 below
            else:
                self.unseen[i] += 1
        self.first = {i: (v[0], v[1], v[2] + dt) for i, v in self.first.items()}
        for i, (zi, si) in meas.items():
            if i in self.filters:
                continue
            if i in self.first and self.first[i][2] > 0:
                z0, s0, age = self.first.pop(i)
                self.filters[i] = IMMGround.from_two_points(
                    z0, s0, zi, si, age, self.settings_for(grp[i])
                )
                self.unseen[i], self.age[i] = 0, 0
                self.nis[i] = deque(maxlen=self.nis_window)
            elif i not in self.first:
                self.first[i] = (zi, si, 0.0)
        for i in [i for i, u in self.unseen.items() if u >= self.max_unseen]:
            self._drop(i)
        self.first = {i: v for i, v in self.first.items() if v[2] < self.max_unseen * dt}
        if not self.filters:
            return {**{k: v.copy() for k, v in EMPTY.items()}, "p_stationary": np.zeros(0)}
        order = list(self.filters)
        sh = np.array([self.filters[i].speed_heading() for i in order]).reshape(-1, 4)
        xs = np.array([self.filters[i].x for i in order])
        return {
            "id": np.array(order),
            "n_m": xs[:, 0],
            "e_m": xs[:, 1],
            "speed_ms": sh[:, 0],
            "speed_sigma_ms": sh[:, 1],
            "heading_deg": sh[:, 2],
            "heading_sigma_deg": sh[:, 3],
            "age": np.array([self.age[i] for i in order], int),
            "nis_mean": np.array([np.mean(self.nis[i]) if self.nis[i] else np.nan for i in order]),
            "nis_n": np.array([len(self.nis[i]) for i in order], int),
            "p_stationary": np.array([self.filters[i].mode[STATIONARY] for i in order]),
        }


class GroupedFilterBank:
    """One filter bank per group (e.g. the IMM for people, constant velocity for vehicles),
    stepped together; same interface and outputs (`p_stationary` NaN where a bank has none).
    Every bank is stepped every frame, so filters without a measurement are predicted and
    dropped as usual."""

    def __init__(self, banks: Mapping[str, TrackFilterBank | IMMFilterBank]) -> None:
        self.banks = dict(banks)

    def step(
        self,
        dt: float,
        ids: NDArray,
        z: NDArray,
        sigma: NDArray,
        groups: Sequence[str] | None = None,
    ) -> dict[str, NDArray]:
        if groups is None:
            raise ValueError("a filter bank per group needs the group of every measurement")
        ids = np.asarray(ids, int)
        z = np.asarray(z, float).reshape(-1, 2)
        cov = as_covariances(sigma, len(z))
        g = np.asarray(list(groups), object)
        unknown = set(g.tolist()) - set(self.banks)
        if unknown:
            raise KeyError(f"no filter bank for groups {sorted(unknown)}")
        outs = []
        for name, bank in self.banks.items():
            m = g == name
            o = bank.step(dt, ids[m], z[m], cov[m], [name] * int(m.sum()))
            if "p_stationary" not in o:
                o = {**o, "p_stationary": np.full(len(o["id"]), np.nan)}
            outs.append(o)
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def make_filter_bank(
    kalman: dict, nis_window: int
) -> TrackFilterBank | IMMFilterBank | GroupedFilterBank:
    """The run configuration's filter: version 1's constant-velocity bank unless `kalman`
    names `motion_model: imm`, whose settings are then read from `kalman["imm"]`: either one
    set of settings, or one per group (`{person: {...}, vehicle: {...}}`). `motion_model` may
    also name a model per group (`{person: imm, vehicle: cv}`)."""
    model = kalman.get("motion_model", "cv")
    if isinstance(model, Mapping):
        banks = {}
        for grp, mm in model.items():
            sub = {**kalman, "motion_model": mm}
            if mm == "imm" and grp in kalman["imm"]:
                sub["imm"] = kalman["imm"][grp]
            banks[grp] = make_filter_bank(sub, nis_window)
        return GroupedFilterBank(banks)
    if model == "cv":
        return TrackFilterBank(
            float(kalman["q"]), int(kalman["max_unseen_frames"]), nis_window=nis_window
        )
    if model == "imm":
        imm = kalman["imm"]
        names = set(IMMSettings.__dataclass_fields__)
        settings: IMMSettings | dict[str, IMMSettings] = (
            IMMSettings.from_mapping(imm)
            if names & set(imm)
            else {g: IMMSettings.from_mapping(v) for g, v in imm.items()}
        )
        return IMMFilterBank(settings, int(kalman["max_unseen_frames"]), nis_window=nis_window)
    raise ValueError(f"unknown motion_model {model!r} (cv or imm)")
