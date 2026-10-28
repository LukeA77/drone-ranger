"""Per-track ground-plane Kalman filters, stepped together (D-612).

Tracks come and go; each frame, every live filter is predicted by `dt` (constant velocity) and
those with a measurement are updated (`GroundKalmanBatch`). A new track is initialised by
two-point differencing on its second measurement. A filter is dropped after `max_unseen` frames
without a measurement (predict only in between).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerialrange.geometry.kalman import GroundKalmanBatch

EMPTY = {
    "id": np.zeros(0, int),
    "n_m": np.zeros(0),
    "e_m": np.zeros(0),
    "speed_ms": np.zeros(0),
    "speed_sigma_ms": np.zeros(0),
}


class TrackFilterBank:
    def __init__(self, q: float, max_unseen: int) -> None:
        self.q, self.max_unseen = q, max_unseen
        self.ids: list[int] = []
        self.bank: GroundKalmanBatch | None = None
        self.unseen: dict[int, int] = {}
        self.first: dict[int, tuple[NDArray, float, float]] = {}  # id -> (z, sigma, age in s)

    def _add(self, i: int, z0: NDArray, s0: float, z1: NDArray, s1: float, dt: float) -> None:
        b = GroundKalmanBatch.from_two_points(
            z0[None], (np.eye(2) * s0**2)[None], z1[None], (np.eye(2) * s1**2)[None], dt, self.q
        )
        if self.bank is None or not self.ids:
            self.bank = b
        else:
            self.bank.x = np.vstack([self.bank.x, b.x])
            self.bank.P = np.concatenate([self.bank.P, b.P])
        self.ids.append(i)
        self.unseen[i] = 0

    def step(self, dt: float, ids: NDArray, z: NDArray, sigma: NDArray) -> dict[str, NDArray]:
        """Advance by `dt` seconds with measurements z (k, 2) of sigma (k,) (metres, isotropic)
        for track ids (k,). Returns id, position, speed and speed sigma of every live filter."""
        ids = np.asarray(ids, int)
        z, sigma = np.asarray(z, float).reshape(-1, 2), np.asarray(sigma, float)
        meas = {
            int(i): (z[j], float(sigma[j])) for j, i in enumerate(ids) if np.isfinite(z[j]).all()
        }
        if self.bank is not None and self.ids:
            self.bank.predict(dt)
            have = np.array([i in meas for i in self.ids])
            Z = np.array([meas[i][0] if i in meas else (0.0, 0.0) for i in self.ids])
            R = np.array([np.eye(2) * (meas[i][1] ** 2 if i in meas else 1.0) for i in self.ids])
            self.bank.update(Z, R, have)
            for i in self.ids:
                self.unseen[i] = 0 if i in meas else self.unseen[i] + 1
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
                    del self.unseen[i]
            self.ids = [self.ids[k] for k in keep]
            self.bank.x, self.bank.P = self.bank.x[keep], self.bank.P[keep]
        self.first = {i: v for i, v in self.first.items() if v[2] < self.max_unseen * dt}
        if not self.ids:
            return {k: v.copy() for k, v in EMPTY.items()}
        sp, sp_sig = self.bank.speed()
        return {
            "id": np.array(self.ids),
            "n_m": self.bank.x[:, 0].copy(),
            "e_m": self.bank.x[:, 1].copy(),
            "speed_ms": sp,
            "speed_sigma_ms": sp_sig,
        }
