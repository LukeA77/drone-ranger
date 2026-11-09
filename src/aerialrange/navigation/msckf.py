"""Multi-state constraint Kalman filter with IMU propagation, camera and GPS updates (D-741).

Method: Mourikis and Roumeliotis (2007), "A Multi-State Constraint Kalman Filter for
Vision-aided Inertial Navigation", ICRA: a window of past IMU poses (clones) is kept in the
state; each feature track is triangulated, its residuals are projected onto the left null space
of the feature Jacobian, gated, and used in one EKF update. Consistency: first-estimate
Jacobians (Huang, Mourikis and Roumeliotis, 2008, ISER; Li and Mourikis, 2013, IJRR 32(6),
"MSCKF 2.0"): the transition blocks that couple attitude to velocity and position use the
propagated (pre-update) estimates, and feature Jacobians use each clone's first estimate.

Frames: world NED, body FRD, camera OpenCV (`aerialrange.geometry.frames`). Attitude error is
in the world frame, R_true = Exp(dtheta) R_est; other errors are additive (true - estimate).

Error state (21 + 6 per clone):
  0:3 attitude, 3:6 position, 6:9 velocity, 9:12 gyro bias, 12:15 accel bias,
  15:18 GPS constant offset, 18:21 GPS Gauss-Markov part, then per clone (attitude, position).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from aerialrange.navigation.alignment import GRAVITY_NED
from aerialrange.navigation.gnss import GnssErrorModel
from aerialrange.navigation.imu import ImuNoise
from aerialrange.navigation.so3 import exp_so3, skew

TH, PO, VE, BG, BA, GC, GM = (slice(3 * i, 3 * i + 3) for i in range(7))
NCORE = 21


@dataclass(frozen=True)
class MsckfSettings:
    """Defaults from OpenVINS's EuRoC configuration (github.com/rpng/open_vins,
    config/euroc_mav/estimator_config.yaml, read 2026-11-08) unless stated."""

    max_clones: int = 11
    max_features_per_update: int = 40  # max_msckf_in_update
    sigma_px: float = 1.0  # up_msckf_sigma_px
    chi2_multiplier: float = 1.0  # up_msckf_chi2_multipler
    gate_prob: float = 0.95  # OpenVINS's chi-squared table is at 95 %
    min_obs: int = 3  # observations needed to use a track
    max_cond: float = 1e4  # OpenVINS feature initialiser max_cond_number
    min_depth_m: float = 0.1  # OpenVINS feature initialiser min_dist
    gn_iterations: int = 5  # OpenVINS feature initialiser max_runs
    fej: bool = True  # use_fej
    use_gps_velocity: bool = True
    # GPS updates after the start; False gives camera and IMU only (GPS used for the alignment
    # alone), for a like-for-like comparison with visual-inertial methods such as OpenVINS
    gps_updates: bool = True
    # Frame the gyroscope's rates (and so its bias and noise) are expressed in. A real gyroscope
    # measures body rates ("body"). Mid-Air's gyroscope is ground-truth angular velocity plus
    # noise and bias, and that angular velocity is in the world frame although the file says
    # "body" (regression on the attitude-derived rates: identity, residual 0.001 rad/s against
    # 0.01 to 0.015 rad/s for body rates; scripts/msckf_imu_frames.py, D-741), so "world".
    gyro_frame: str = "body"


@dataclass
class Clone:
    t: float
    R: NDArray[np.float64]
    p: NDArray[np.float64]
    R_fej: NDArray[np.float64]
    p_fej: NDArray[np.float64]


@dataclass
class UpdateStats:
    features_tried: int = 0
    features_used: int = 0
    rejected_chi2: int = 0
    rejected_triangulation: int = 0
    rows: int = 0
    nis: list[float] = field(default_factory=list)


class Msckf:
    def __init__(
        self,
        R_bc: NDArray,
        t_bc: NDArray,
        focal_px: float,
        settings: MsckfSettings,
        noise: ImuNoise,
        gnss: GnssErrorModel,
        t0: float,
        R: NDArray,
        p: NDArray,
        v: NDArray,
        bg: NDArray,
        ba: NDArray,
        P0: NDArray,
    ) -> None:
        self.R_bc = np.asarray(R_bc, float)
        self.t_bc = np.asarray(t_bc, float)
        self.focal = float(focal_px)
        self.s = settings
        self.noise = noise
        self.gnss = gnss
        self.t = float(t0)
        self.R = np.asarray(R, float).copy()
        self.p = np.asarray(p, float).copy()
        self.v = np.asarray(v, float).copy()
        self.bg = np.asarray(bg, float).copy()
        self.ba = np.asarray(ba, float).copy()
        self.gc = np.zeros(3)
        self.gm = np.zeros(3)
        self.P = np.asarray(P0, float).copy()
        if self.P.shape != (NCORE, NCORE):
            raise ValueError(f"P0 must be {NCORE}x{NCORE}")
        self.clones: dict[int, Clone] = {}
        self._fej_p = self.p.copy()
        self._fej_v = self.v.copy()

    # ------------------------------------------------------------------ propagation
    def propagate(self, gyr: NDArray, acc: NDArray, dt: float) -> None:
        """Propagate through IMU samples (each the mean rate over its `dt` step: pass the
        average of consecutive raw samples), state and covariance."""
        world_gyro = self.s.gyro_frame == "world"
        if self.s.gyro_frame not in ("body", "world"):
            raise ValueError(f"gyro_frame must be 'body' or 'world', got {self.s.gyro_frame!r}")
        Phi_tot = np.eye(NCORE)
        Q_tot = np.zeros((NCORE, NCORE))
        n = self.noise
        a_gm = float(np.exp(-dt / self.gnss.gm_tau_s))
        for w_m, f_m in zip(np.asarray(gyr, float), np.asarray(acc, float), strict=True):
            w = w_m - self.bg
            f = f_m - self.ba
            R0, p0, v0 = self.R, self.p, self.v
            if world_gyro:  # dR/dt = [w_world]x R
                R_mid = exp_so3(w * dt / 2) @ R0
                R1 = exp_so3(w * dt) @ R0
            else:  # dR/dt = R [w_body]x
                R_mid = R0 @ exp_so3(w * dt / 2)
                R1 = R0 @ exp_so3(w * dt)
            a_w = R_mid @ f + GRAVITY_NED
            v1 = v0 + a_w * dt
            p1 = p0 + v0 * dt + 0.5 * a_w * dt**2
            # error-state transition (first order in dt plus the second-order terms)
            Phi = np.eye(NCORE)
            if self.s.fej:
                pf, vf = self._fej_p, self._fej_v
            else:
                pf, vf = p0, v0
            Phi[VE, TH] = -skew(v1 - vf - GRAVITY_NED * dt)
            Phi[PO, TH] = -skew(p1 - pf - vf * dt - 0.5 * GRAVITY_NED * dt**2)
            Phi[PO, VE] = np.eye(3) * dt
            # gyro bias (and noise) enter the world-frame attitude error through G
            G = np.eye(3) if world_gyro else R_mid
            if world_gyro:  # d(dtheta)/dt = [w]x dtheta - dbg - n_g
                Phi[TH, TH] = exp_so3(w * dt)
            Phi[TH, BG] = -G * dt
            Phi[VE, BA] = -R_mid * dt
            Phi[PO, BA] = -0.5 * R_mid * dt**2
            Phi[VE, BG] = 0.5 * skew(R_mid @ f) @ G * dt**2
            Phi[PO, BG] = skew(R_mid @ f) @ G * dt**3 / 6.0
            Phi[GM, GM] = np.eye(3) * a_gm
            Q = np.zeros((NCORE, NCORE))
            Cg = G @ np.diag(n.gyr**2) @ G.T
            Ca = R_mid @ np.diag(n.acc**2) @ R_mid.T
            Q[TH, TH] = Cg * dt**2
            Q[VE, VE] = Ca * dt**2
            Q[PO, PO] = Ca * dt**4 / 4.0
            Q[PO, VE] = Ca * dt**3 / 2.0
            Q[VE, PO] = Q[PO, VE].T
            Q[BG, BG] = np.diag(n.gyr_bias_rw**2)
            Q[BA, BA] = np.diag(n.acc_bias_rw**2)
            Q[GM, GM] = np.diag(self.gnss.gm_sigma**2) * (1.0 - a_gm**2)
            Phi_tot = Phi @ Phi_tot
            Q_tot = Phi @ Q_tot @ Phi.T + Q
            self.R, self.p, self.v = _orthonormal(R1), p1, v1
            self.gm = self.gm * a_gm
            self._fej_p, self._fej_v = p1.copy(), v1.copy()
            self.t += dt
        self.last_phi = Phi_tot
        P = self.P
        P[:NCORE, :NCORE] = Phi_tot @ P[:NCORE, :NCORE] @ Phi_tot.T + Q_tot
        if P.shape[0] > NCORE:
            P[:NCORE, NCORE:] = Phi_tot @ P[:NCORE, NCORE:]
            P[NCORE:, :NCORE] = P[:NCORE, NCORE:].T
        self.P = 0.5 * (P + P.T)

    # ------------------------------------------------------------------ clones
    def augment(self, frame_id: int) -> None:
        """Add the current IMU pose as a clone (its first estimate is this propagated value)."""
        self.clones[frame_id] = Clone(
            self.t, self.R.copy(), self.p.copy(), self.R.copy(), self.p.copy()
        )
        N = self.P.shape[0]
        J = np.zeros((6, N))
        J[0:3, TH] = np.eye(3)
        J[3:6, PO] = np.eye(3)
        P = np.zeros((N + 6, N + 6))
        P[:N, :N] = self.P
        P[N:, :N] = J @ self.P
        P[:N, N:] = P[N:, :N].T
        P[N:, N:] = J @ self.P @ J.T
        self.P = P

    def clone_index(self, frame_id: int) -> int:
        return NCORE + 6 * list(self.clones).index(frame_id)

    def marginalize(self, frame_id: int) -> None:
        i = self.clone_index(frame_id)
        keep = np.r_[0:i, i + 6 : self.P.shape[0]]
        self.P = self.P[np.ix_(keep, keep)]
        del self.clones[frame_id]

    def oldest_clone(self) -> int:
        return next(iter(self.clones))

    # ------------------------------------------------------------------ features
    def _cam_pose(self, c: Clone, fej: bool) -> tuple[NDArray, NDArray]:
        R, p = (c.R_fej, c.p_fej) if fej else (c.R, c.p)
        return R @ self.R_bc, p + R @ self.t_bc

    def triangulate(self, obs: list[tuple[int, NDArray]]) -> NDArray | None:
        """World point from normalised observations [(clone id, (x, y))]; None if ill posed."""
        A = np.zeros((3, 3))
        b = np.zeros(3)
        poses = []
        for fid, xy in obs:
            Rwc, pwc = self._cam_pose(self.clones[fid], fej=False)
            d = Rwc @ np.array([xy[0], xy[1], 1.0])
            d /= np.linalg.norm(d)
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ pwc
            poses.append((Rwc, pwc, xy))
        if np.linalg.cond(A) > self.s.max_cond:
            return None
        X = np.linalg.solve(A, b)
        for _ in range(self.s.gn_iterations):  # Gauss-Newton on the reprojection error
            H, r = [], []
            for Rwc, pwc, xy in poses:
                Pc = Rwc.T @ (X - pwc)
                if Pc[2] <= self.s.min_depth_m:
                    return None
                Jp = np.array([[1, 0, -Pc[0] / Pc[2]], [0, 1, -Pc[1] / Pc[2]]]) / Pc[2]
                H.append(Jp @ Rwc.T)
                r.append(xy - Pc[:2] / Pc[2])
            H = np.vstack(H)
            r = np.concatenate(r)
            dx = np.linalg.lstsq(H, r, rcond=None)[0]
            X = X + dx
            if np.linalg.norm(dx) < 1e-6 * max(1.0, np.linalg.norm(X)):
                break
        for Rwc, pwc, _ in poses:
            if (Rwc.T @ (X - pwc))[2] <= self.s.min_depth_m:
                return None
        return X

    def feature_rows(
        self, obs: list[tuple[int, NDArray]], stats: UpdateStats
    ) -> tuple[NDArray, NDArray] | None:
        """Null-space projected, gated residual rows and Jacobian for one track (or None)."""
        obs = [(fid, xy) for fid, xy in obs if fid in self.clones]
        if len(obs) < self.s.min_obs:
            return None
        X = self.triangulate(obs)
        if X is None:
            stats.rejected_triangulation += 1
            return None
        N = self.P.shape[0]
        m = 2 * len(obs)
        Hx = np.zeros((m, N))
        Hf = np.zeros((m, 3))
        r = np.zeros(m)
        for j, (fid, xy) in enumerate(obs):
            c = self.clones[fid]
            z, _, _, _ = camera_projection(c.R, c.p, self.R_bc, self.t_bc, X)
            r[2 * j : 2 * j + 2] = xy - z
            Rj, pj = (c.R_fej, c.p_fej) if self.s.fej else (c.R, c.p)
            _, Jth, Jpo, Jx = camera_projection(Rj, pj, self.R_bc, self.t_bc, X)
            i = self.clone_index(fid)
            Hx[2 * j : 2 * j + 2, i : i + 3] = Jth
            Hx[2 * j : 2 * j + 2, i + 3 : i + 6] = Jpo
            Hf[2 * j : 2 * j + 2] = Jx
        Q, _ = np.linalg.qr(Hf, mode="complete")
        A = Q[:, 3:]
        H0 = A.T @ Hx
        r0 = A.T @ r
        sig2 = (self.s.sigma_px / self.focal) ** 2
        S = H0 @ self.P @ H0.T + sig2 * np.eye(len(r0))
        nis = float(r0 @ np.linalg.solve(S, r0))
        stats.features_tried += 1
        if nis > self.s.chi2_multiplier * chi2.ppf(self.s.gate_prob, len(r0)):
            stats.rejected_chi2 += 1
            return None
        stats.nis.append(nis / len(r0))
        stats.features_used += 1
        return H0, r0

    def update_features(self, tracks: list[list[tuple[int, NDArray]]]) -> UpdateStats:
        """One EKF update from several feature tracks (longest first, capped per settings)."""
        stats = UpdateStats()
        tracks = sorted(tracks, key=len, reverse=True)[: self.s.max_features_per_update]
        Hs, rs = [], []
        for obs in tracks:
            out = self.feature_rows(obs, stats)
            if out is not None:
                Hs.append(out[0])
                rs.append(out[1])
        if not Hs:
            return stats
        H = np.vstack(Hs)
        r = np.concatenate(rs)
        if H.shape[0] > H.shape[1]:  # measurement compression (QR)
            Qh, Rh = np.linalg.qr(H)
            H, r = Rh, Qh.T @ r
        stats.rows = len(r)
        sig2 = (self.s.sigma_px / self.focal) ** 2
        self._ekf_update(H, r, sig2 * np.eye(len(r)))
        return stats

    # ------------------------------------------------------------------ GPS
    def update_gps(self, pos: NDArray, vel: NDArray | None) -> float:
        """GPS position (and velocity) update; returns the normalised innovation squared."""
        N = self.P.shape[0]
        rows_H, rows_r, rows_R = [], [], []
        H = np.zeros((3, N))
        H[:, PO] = np.eye(3)
        H[:, GC] = np.eye(3)
        H[:, GM] = np.eye(3)
        rows_H.append(H)
        rows_r.append(np.asarray(pos, float) - (self.p + self.gc + self.gm))
        rows_R.append(np.diag(self.gnss.white_sigma**2))
        if vel is not None and self.s.use_gps_velocity:
            Hv = np.zeros((3, N))
            Hv[:, VE] = np.eye(3)
            rows_H.append(Hv)
            rows_r.append(np.asarray(vel, float) - self.v)
            rows_R.append(np.eye(3) * self.gnss.vel_sigma**2)
        H = np.vstack(rows_H)
        r = np.concatenate(rows_r)
        Rm = _block_diag(rows_R)
        S = H @ self.P @ H.T + Rm
        nis = float(r @ np.linalg.solve(S, r))
        self._ekf_update(H, r, Rm)
        return nis

    # ------------------------------------------------------------------ EKF core
    def _ekf_update(self, H: NDArray, r: NDArray, Rm: NDArray) -> None:
        P = self.P
        S = H @ P @ H.T + Rm
        K = np.linalg.solve(S, H @ P).T
        dx = K @ r
        IKH = np.eye(P.shape[0]) - K @ H
        self.P = IKH @ P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._apply(dx)

    def _apply(self, dx: NDArray) -> None:
        self.R = _orthonormal(exp_so3(dx[TH]) @ self.R)
        self.p = self.p + dx[PO]
        self.v = self.v + dx[VE]
        self.bg = self.bg + dx[BG]
        self.ba = self.ba + dx[BA]
        self.gc = self.gc + dx[GC]
        self.gm = self.gm + dx[GM]
        for k, c in enumerate(self.clones.values()):
            i = NCORE + 6 * k
            c.R = _orthonormal(exp_so3(dx[i : i + 3]) @ c.R)
            c.p = c.p + dx[i + 3 : i + 6]


def camera_projection(
    R_wb: NDArray, p_wb: NDArray, R_bc: NDArray, t_bc: NDArray, X: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Normalised image point of world point X from body pose (R_wb, p_wb), and its Jacobians
    with respect to the world-frame attitude error, the body position and X."""
    Rcw = (R_wb @ R_bc).T
    Pc = Rcw @ (X - p_wb) - R_bc.T @ t_bc
    Jp = np.array([[1, 0, -Pc[0] / Pc[2]], [0, 1, -Pc[1] / Pc[2]]]) / Pc[2]
    return Pc[:2] / Pc[2], Jp @ Rcw @ skew(X - p_wb), -Jp @ Rcw, Jp @ Rcw


def _orthonormal(R: NDArray) -> NDArray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _block_diag(blocks: list[NDArray]) -> NDArray:
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        out[i : i + k, i : i + k] = b
        i += k
    return out
