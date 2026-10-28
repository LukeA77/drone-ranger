"""Ground-plane constant-velocity Kalman filter.

State `x = [north, east, v_north, v_east]` in the world frame (metres, metres per second), so
the drone's own motion does not leak into target velocity. Measurements are per-frame ground
points with their own 2x2 covariance from the uncertainty model (heteroscedastic).

Model: continuous white-noise acceleration with spectral density `q` (m^2/s^3), the
`{{KALMAN_Q}}` placeholder. Initialisation: two-point differencing. Both follow Bar-Shalom,
Li and Kirubarajan (2001), Estimation with Applications to Tracking and Navigation, Wiley,
the standard constant-velocity model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

I2 = np.eye(2)


def cv_transition(dt: float) -> NDArray[np.float64]:
    """State transition over `dt` seconds."""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    return np.block([[I2, dt * I2], [np.zeros((2, 2)), I2]])


def cv_process_noise(dt: float, q: float) -> NDArray[np.float64]:
    """Process-noise covariance over `dt` seconds for spectral density `q` (m^2/s^3)."""
    if dt < 0 or q < 0:
        raise ValueError("dt and q must be non-negative")
    return q * np.block([[dt**3 / 3 * I2, dt**2 / 2 * I2], [dt**2 / 2 * I2, dt * I2]])


@dataclass
class GroundKalman:
    """Filter state `x` (4,), covariance `P` (4, 4) and process-noise density `q`."""

    x: NDArray[np.float64]
    P: NDArray[np.float64]
    q: float

    @classmethod
    def from_two_points(
        cls, z1: ArrayLike, R1: ArrayLike, z2: ArrayLike, R2: ArrayLike, dt: float, q: float
    ) -> GroundKalman:
        """Initialise at the second measurement with velocity (z2 - z1) / dt."""
        if dt <= 0:
            raise ValueError("dt must be positive for two-point initialisation")
        z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
        R1, R2 = np.asarray(R1, float), np.asarray(R2, float)
        x = np.concatenate([z2, (z2 - z1) / dt])
        P = np.block([[R2, R2 / dt], [R2 / dt, (R1 + R2) / dt**2]])
        return cls(x, P, q)

    def predict(self, dt: float) -> None:
        """Propagate the state by `dt` seconds."""
        F = cv_transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + cv_process_noise(dt, self.q)

    def update(self, z: ArrayLike, R: ArrayLike) -> None:
        """Fuse a ground-point measurement `z` (north, east) with covariance `R` (m^2)."""
        H = np.hstack([I2, np.zeros((2, 2))])
        S = H @ self.P @ H.T + np.asarray(R, float)
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ (np.asarray(z, float) - H @ self.x)
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ np.asarray(R, float) @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def position(self) -> NDArray[np.float64]:
        return self.x[:2]

    @property
    def velocity(self) -> NDArray[np.float64]:
        return self.x[2:]

    def speed_heading(self) -> tuple[float, float, float, float]:
        """(speed m/s, speed sigma, heading degrees clockwise from North, heading sigma).

        First-order sigmas from the velocity covariance; heading is NaN at zero speed.
        """
        v, Pv = self.x[2:], self.P[2:, 2:]
        s = float(np.hypot(*v))
        if s == 0:
            return 0.0, float(np.sqrt(np.trace(Pv) / 2)), float("nan"), float("nan")
        g_s = v / s
        g_h = np.array([-v[1], v[0]]) / s**2
        heading = float(np.degrees(np.arctan2(v[1], v[0]))) % 360.0
        return (
            s,
            float(np.sqrt(g_s @ Pv @ g_s)),
            heading,
            float(np.degrees(np.sqrt(g_h @ Pv @ g_h))),
        )


def run_filter(
    dts: Sequence[float],
    zs: Sequence[ArrayLike | None],
    Rs: Sequence[ArrayLike | None],
    q: float,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Filter a sequence of measurements. `zs[k]` None means a missed detection
    (predict only). `dts[k]` is the time from measurement k to k + 1, so
    `len(zs) == len(dts) + 1`. The first two measurements initialise the filter.

    Returns (x, P) after each of measurements 1 .. n-1.
    """
    if len(zs) != len(dts) + 1 or len(Rs) != len(zs):
        raise ValueError("need len(zs) == len(Rs) == len(dts) + 1")
    if zs[0] is None or zs[1] is None:
        raise ValueError("the first two measurements must be present")
    kf = GroundKalman.from_two_points(zs[0], Rs[0], zs[1], Rs[1], dts[0], q)
    out = [(kf.x.copy(), kf.P.copy())]
    for k in range(2, len(zs)):
        kf.predict(dts[k - 1])
        if zs[k] is not None:
            kf.update(zs[k], Rs[k])
        out.append((kf.x.copy(), kf.P.copy()))
    return out


@dataclass
class GroundKalmanBatch:
    """Many independent `GroundKalman` filters stepped together with array operations (same
    model and equations; Phase 5 speed, D-612). `x` (n, 4), `P` (n, 4, 4), shared `q`."""

    x: NDArray[np.float64]
    P: NDArray[np.float64]
    q: float

    @classmethod
    def from_two_points(
        cls, z1: ArrayLike, R1: ArrayLike, z2: ArrayLike, R2: ArrayLike, dt: float, q: float
    ) -> GroundKalmanBatch:
        """Two-point initialisation for each filter: z (n, 2), R (n, 2, 2)."""
        if dt <= 0:
            raise ValueError("dt must be positive for two-point initialisation")
        z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
        R1, R2 = np.asarray(R1, float), np.asarray(R2, float)
        x = np.concatenate([z2, (z2 - z1) / dt], axis=1)
        P = np.concatenate(
            [
                np.concatenate([R2, R2 / dt], axis=2),
                np.concatenate([R2 / dt, (R1 + R2) / dt**2], axis=2),
            ],
            axis=1,
        )
        return cls(x, P, q)

    def predict(self, dt: float) -> None:
        F = cv_transition(dt)
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + cv_process_noise(dt, self.q)

    def update(self, z: ArrayLike, R: ArrayLike, measured: ArrayLike | None = None) -> None:
        """Fuse measurements z (n, 2) with covariances R (n, 2, 2); filters where `measured`
        is False are left predicted only (a missed detection)."""
        z, R = np.asarray(z, float), np.asarray(R, float)
        m = np.ones(len(self.x), bool) if measured is None else np.asarray(measured, bool)
        if not m.any():
            return
        x, P, zm, Rm = self.x[m], self.P[m], z[m], R[m]
        S = P[:, :2, :2] + Rm
        K = np.linalg.solve(S, P[:, :2, :]).transpose(0, 2, 1)  # (k, 4, 2)
        x = x + np.einsum("kij,kj->ki", K, zm - x[:, :2])
        H = np.hstack([I2, np.zeros((2, 2))])
        IKH = np.eye(4) - K @ H
        P = IKH @ P @ IKH.transpose(0, 2, 1) + K @ Rm @ K.transpose(0, 2, 1)  # Joseph form
        self.x[m], self.P[m] = x, 0.5 * (P + P.transpose(0, 2, 1))

    def speed(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Speed (m/s) and its first-order sigma for every filter."""
        v, Pv = self.x[:, 2:], self.P[:, 2:, 2:]
        s = np.hypot(v[:, 0], v[:, 1])
        g = np.where(s[:, None] > 0, v / np.where(s > 0, s, 1.0)[:, None], 0.0)
        var = np.einsum("ki,kij,kj->k", g, Pv, g)
        var = np.where(s > 0, var, np.trace(Pv, axis1=1, axis2=2) / 2)
        return s, np.sqrt(var)
