"""Interacting multiple model (IMM) ground filter: stationary, constant velocity, coordinated turn.

Version 1 tracks every object with one constant-velocity filter (`geometry/kalman.py`). Parked
cars and standing people then show speeds made of measurement noise, and turning vehicles lag.
This filter runs three motion models side by side and weights them by how well each explains
the measurements (Blom and Bar-Shalom (1988), IEEE Transactions on Automatic Control 33(8);
the cycle as in Bar-Shalom, Li and Kirubarajan (2001), Estimation with Applications to Tracking
and Navigation, Wiley, Section 11.6):

1. **Stationary:** the object does not move; its position may wander slowly (random walk with
   density `q_stationary`, m^2/s). Velocity and turn rate are held at zero.
2. **Constant velocity:** the version 1 model (white-noise acceleration, density `q_cv`).
3. **Coordinated turn:** constant speed and constant turn rate `w` (rad/s, positive turning
   from North towards East), with white-noise acceleration `q_ct` and turn-rate noise `q_turn`
   (rad^2/s^3); nonlinear, so propagated as an extended Kalman filter (same book, Section 11.7).

All three share one state, `x = [north, east, v_north, v_east, w]`, so they can be mixed.

**Mode switching** is a continuous-time Markov chain with rates `switch_rates[i][j]` (per
second, from mode i to mode j; the diagonal is ignored), so the per-step switching matrix is
expm(L dt) for the generator L and the same settings hold at any frame rate (SynDrone 25 Hz,
AU-AIR 5 Hz, the pipeline's detection stride).

**Gating:** a measurement whose normalised innovation squared against the combined prediction
exceeds the chi-square quantile at `gate_prob` (2 degrees of freedom) is not used. Seen from
the filter, an identity switch is a run of such measurements: after `reset_after` rejections in
a row `needs_reset` is set and the caller restarts the track from two new measurements, as a new
track starts. Every setting is required: none has a default (they are fitted on development
data).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from aerialrange.geometry.kalman import GroundKalman, cv_process_noise

N_MODES = 3
STATIONARY, CV, CT = range(N_MODES)
MODE_NAMES = ("stationary", "constant_velocity", "coordinated_turn")
H = np.hstack([np.eye(2), np.zeros((2, 3))])
SMALL_TURN = 1e-4  # rad/s; below this the turn coefficients use their Taylor series


@dataclass(frozen=True)
class IMMSettings:
    q_stationary: float  # m^2/s, position random walk of a stationary object
    q_cv: float  # m^2/s^3, constant-velocity model acceleration density
    q_ct: float  # m^2/s^3, coordinated-turn model acceleration density
    q_turn: float  # rad^2/s^3, turn-rate noise density
    turn_rate_sigma0: float  # rad/s, turn-rate sigma at initialisation
    switch_rates: tuple[tuple[float, ...], ...]  # 1/s, mode i to mode j (diagonal ignored)
    mode0: tuple[float, ...]  # prior mode probabilities (long-run shares)
    velocity_sigma_moving: float  # m/s per axis, spread of true velocities of moving objects
    gate_prob: float  # gate: chi-square quantile (2 degrees of freedom)
    reset_after: int  # rejected measurements in a row that mean a reset
    _generator: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _cache: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        r = np.asarray(self.switch_rates, float)
        if r.shape != (N_MODES, N_MODES):
            raise ValueError("switch_rates must be a 3x3 matrix")
        r = r * (1 - np.eye(N_MODES))
        if (r < 0).any():
            raise ValueError("switch rates must be non-negative")
        m0 = np.asarray(self.mode0, float)
        if m0.shape != (N_MODES,) or (m0 < 0).any() or not np.isclose(m0.sum(), 1):
            raise ValueError("mode0 must be 3 probabilities summing to 1")
        noise = (
            self.q_stationary,
            self.q_cv,
            self.q_ct,
            self.q_turn,
            self.turn_rate_sigma0,
            self.velocity_sigma_moving,
        )
        if min(noise) < 0:
            raise ValueError("noise densities and sigmas must be non-negative")
        if not 0 < self.gate_prob < 1 or self.reset_after < 1:
            raise ValueError("gate_prob must be in (0, 1) and reset_after at least 1")
        object.__setattr__(self, "_generator", r - np.diag(r.sum(1)))
        object.__setattr__(self, "_cache", {})

    def switch_matrix(self, dt: float) -> NDArray[np.float64]:
        """Per-step switching probabilities over `dt` seconds, expm(L dt); rows sum to 1."""
        key = round(float(dt), 9)
        m = self._cache.get(key)
        if m is None:
            m = expm(self._generator * max(float(dt), 0.0))
            m = np.clip(m, 0.0, None)
            m /= m.sum(1, keepdims=True)
            if len(self._cache) < 256:
                self._cache[key] = m
        return m

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> IMMSettings:
        return cls(**{k: m[k] for k in cls.__dataclass_fields__ if not k.startswith("_")})


def gate_threshold(prob: float) -> float:
    """Chi-square quantile with 2 degrees of freedom: P(chi2 <= t) = 1 - exp(-t / 2)."""
    return float(-2.0 * np.log1p(-prob))


def _turn_coeffs(w: float, dt: float) -> tuple[float, float, float, float]:
    """a = sin(w dt) / w, b = (1 - cos(w dt)) / w and their derivatives in w."""
    if abs(w) < SMALL_TURN:
        a = dt - w**2 * dt**3 / 6
        b = w * dt**2 / 2 - w**3 * dt**4 / 24
        return a, b, -w * dt**3 / 3, dt**2 / 2 - w**2 * dt**4 / 8
    s, c = np.sin(w * dt), np.cos(w * dt)
    return s / w, (1 - c) / w, (dt * c * w - s) / w**2, (dt * s * w - (1 - c)) / w**2


def ct_transition(x: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Coordinated-turn propagation of `x = [n, e, vn, ve, w]` over `dt` seconds (exact for a
    circle at constant speed; constant velocity at w = 0)."""
    n, e, vn, ve, w = np.asarray(x, float)
    a, b, _, _ = _turn_coeffs(w, dt)
    s, c = np.sin(w * dt), np.cos(w * dt)
    return np.array([n + a * vn - b * ve, e + b * vn + a * ve, c * vn - s * ve, s * vn + c * ve, w])


def ct_jacobian(x: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Jacobian of `ct_transition` with respect to the state."""
    _, _, vn, ve, w = np.asarray(x, float)
    a, b, da, db = _turn_coeffs(w, dt)
    s, c = np.sin(w * dt), np.cos(w * dt)
    return np.array(
        [
            [1, 0, a, -b, da * vn - db * ve],
            [0, 1, b, a, db * vn + da * ve],
            [0, 0, c, -s, -dt * (s * vn + c * ve)],
            [0, 0, s, c, dt * (c * vn - s * ve)],
            [0, 0, 0, 0, 1],
        ],
        float,
    )


def _model_predict(
    j: int, x: NDArray, P: NDArray, dt: float, s: IMMSettings
) -> tuple[NDArray, NDArray]:
    Q = np.zeros((5, 5))
    if j == STATIONARY:
        F = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
        Q[:2, :2] = s.q_stationary * dt * np.eye(2)
        return F @ x, F @ P @ F.T + Q
    if j == CV:
        F = np.eye(5)
        F[0, 2] = F[1, 3] = dt
        F[4, 4] = 0.0
        Q[:4, :4] = cv_process_noise(dt, s.q_cv)
        return F @ x, F @ P @ F.T + Q
    F = ct_jacobian(x, dt)
    Q[:4, :4] = cv_process_noise(dt, s.q_ct)
    Q[4, 4] = s.q_turn * dt
    return ct_transition(x, dt), F @ P @ F.T + Q


def _combine(w: NDArray, xs: NDArray, Ps: NDArray) -> tuple[NDArray, NDArray]:
    """Moment-matched Gaussian of a mixture with weights w (m,), means (m, 5), covs (m, 5, 5)."""
    x = w @ xs
    d = xs - x
    P = np.einsum("m,mij->ij", w, Ps) + np.einsum("m,mi,mj->ij", w, d, d)
    return x, 0.5 * (P + P.T)


def _log_normal2(v: NDArray, C: NDArray) -> float:
    _, logdet = np.linalg.slogdet(C)
    return float(-0.5 * (v @ np.linalg.solve(C, v) + logdet) - np.log(2 * np.pi))


def initial_modes(
    v: ArrayLike, R1: ArrayLike, R2: ArrayLike, dt: float, s: IMMSettings
) -> NDArray[np.float64]:
    """Mode probabilities after seeing the two-point velocity `v` (see `from_two_points`)."""
    v = np.asarray(v, float)
    Cv = (np.asarray(R1, float) + np.asarray(R2, float)) / dt**2
    still = Cv + np.eye(2) * s.q_stationary / dt
    moving = Cv + np.eye(2) * s.velocity_sigma_moving**2
    log_l = np.array([_log_normal2(v, still), _log_normal2(v, moving), _log_normal2(v, moving)])
    prior = np.asarray(s.mode0, float)
    logw = log_l + np.log(np.maximum(prior, 1e-300))
    w = np.exp(logw - logw.max()) * (prior > 0)
    return w / w.sum()


class IMMGround:
    """One track's IMM filter. `x`, `P`: combined estimate; `mode`: model probabilities."""

    def __init__(self, xs: NDArray, Ps: NDArray, mode: NDArray, settings: IMMSettings) -> None:
        self.xs, self.Ps, self.mode, self.s = xs, Ps, mode, settings
        self.gate = gate_threshold(settings.gate_prob)
        self.rejections = 0
        self.nis = float("nan")

    @classmethod
    def from_two_points(
        cls,
        z1: ArrayLike,
        R1: ArrayLike,
        z2: ArrayLike,
        R2: ArrayLike,
        dt: float,
        settings: IMMSettings,
    ) -> IMMGround:
        """Two-point initialisation as in version 1, turn rate 0 with `turn_rate_sigma0`.

        The starting mode probabilities are the posterior given the two-point velocity v,
        prior `mode0`: under the stationary model v is measurement noise (covariance
        (R1 + R2) / dt^2, plus the stationary random walk over dt), under the moving models it
        also carries the spread of true velocities (`velocity_sigma_moving` per axis); the
        two moving models keep their prior ratio."""
        kf = GroundKalman.from_two_points(z1, R1, z2, R2, dt, settings.q_cv)
        x = np.append(kf.x, 0.0)
        P = np.zeros((5, 5))
        P[:4, :4] = kf.P
        P[4, 4] = settings.turn_rate_sigma0**2
        xs = np.repeat(x[None], N_MODES, axis=0)
        Ps = np.repeat(P[None], N_MODES, axis=0)
        return cls(xs, Ps, initial_modes(kf.x[2:4], R1, R2, dt, settings), settings)

    @property
    def x(self) -> NDArray[np.float64]:
        return self.mode @ self.xs

    @property
    def P(self) -> NDArray[np.float64]:
        return _combine(self.mode, self.xs, self.Ps)[1]

    @property
    def needs_reset(self) -> bool:
        return self.rejections >= self.s.reset_after

    def predict(self, dt: float) -> None:
        """Mix the model estimates by the switching matrix, then propagate each model."""
        sw = self.s.switch_matrix(dt)
        c = self.mode @ sw  # predicted mode probabilities
        mix = sw * self.mode[:, None] / np.where(c > 0, c, 1.0)[None, :]  # mix[i, j] = mu_i|j
        xs, Ps = np.empty_like(self.xs), np.empty_like(self.Ps)
        for j in range(N_MODES):
            x0, P0 = _combine(mix[:, j], self.xs, self.Ps)
            xs[j], Ps[j] = _model_predict(j, x0, P0, dt, self.s)
        self.xs, self.Ps, self.mode = xs, Ps, c

    def update(self, z: ArrayLike, R: ArrayLike) -> bool:
        """Fuse a ground point `z` (north, east) with covariance `R` (m^2). Returns False, and
        changes nothing, if the measurement falls outside the gate."""
        z, R = np.asarray(z, float), np.asarray(R, float)
        xc, Pc = _combine(self.mode, self.xs, self.Ps)
        nu = z - xc[:2]
        nis = float(nu @ np.linalg.solve(Pc[:2, :2] + R, nu))
        if nis > self.gate:
            self.rejections += 1
            return False
        self.nis = nis
        self.rejections = 0
        log_like = np.empty(N_MODES)
        for j in range(N_MODES):
            x, P = self.xs[j], self.Ps[j]
            S = P[:2, :2] + R
            v = z - x[:2]
            K = np.linalg.solve(S, P[:2, :]).T
            self.xs[j] = x + K @ v
            IKH = np.eye(5) - K @ H
            Pn = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form
            self.Ps[j] = 0.5 * (Pn + Pn.T)
            _, logdet = np.linalg.slogdet(S)
            log_like[j] = -0.5 * (v @ np.linalg.solve(S, v) + logdet) - np.log(2 * np.pi)
        logw = log_like + np.log(np.maximum(self.mode, 1e-300))
        w = np.exp(logw - logw.max())
        self.mode = w / w.sum()
        return True

    def speed_heading(self) -> tuple[float, float, float, float]:
        """(speed m/s, sigma, heading degrees clockwise from North, sigma), as version 1."""
        x, P = self.x, self.P
        return GroundKalman(x[:4], P[:4, :4], self.s.q_cv).speed_heading()
