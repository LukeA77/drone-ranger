"""In-flight attitude alignment from IMU and GPS velocity (D-741).

Mid-Air flights start already moving (5 to 20 m/s), and there is no magnetometer, so the filter
cannot start from a static levelling. Over each 1 s interval between GPS velocity fixes,
    v(t_b) - v(t_a) - g (t_b - t_a) = R_0 * sum_k R_rel(t_k) (f_k - b_a) dt,
where R_rel is the attitude relative to t_0 integrated from the gyroscope and f the
accelerometer samples. Each interval gives one vector pair (world w_i, body-at-t0 u_i);
R_0 solves Wahba's problem (Wahba, 1965, SIAM Review 7(3)) by singular value decomposition
(Markley, 1988, J. Astronautical Sciences 36(3)). Gravity dominates every pair, so yaw is
determined only by horizontal acceleration; the attitude covariance from the linearised fit
(Fisher information sum [a_i]x^T [a_i]x / s_i^2, a_i = R_0 u_i) says how well, and the window
grows until it is good enough (`align`).

With a world-frame gyroscope (Mid-Air, see `MsckfSettings.gyro_frame`) the attitude path is
R(t) = A(t) R_0 instead of R_0 A(t), so R_0 sits inside the sum: the Wahba solution with A at
each interval's centre starts a Gauss-Newton refinement on the exact sums (both frames are
refined the same way; for body rates the start is already the exact minimum).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerialrange.navigation.so3 import exp_so3, skew

GRAVITY_NED = np.array([0.0, 0.0, 9.81])  # m/s^2; the value Mid-Air's IMU generator uses


@dataclass(frozen=True)
class VisionPair:
    """Travel direction seen by the camera between two frames (`navigation.heading`) and the
    same direction from GPS velocity, for the alignment. Directions are unit vectors; sigmas
    are angular standard deviations (radians, as RMS angle) of each."""

    t_s: float  # time of the first frame from the window start
    dt_s: float  # time between the two frames
    d_c: NDArray[np.float64]  # camera frame at the first frame
    sigma_c: float
    d_w: NDArray[np.float64]  # world frame
    sigma_w: float


@dataclass(frozen=True)
class Alignment:
    """Result of the alignment. Joint error state x = (attitude, gyro bias, accel bias): world-
    frame attitude error, biases additive (true - estimate)."""

    R0: NDArray[np.float64]  # body-to-world attitude at the window start
    bg: NDArray[np.float64]  # gyro bias over the window
    ba: NDArray[np.float64]  # accel bias over the window
    cov: NDArray[np.float64]  # 9x9 covariance of x at the window start
    R_end: NDArray[np.float64]  # attitude at the window end
    cov_end: NDArray[np.float64]  # 9x9 covariance of x at the window end
    window_s: float
    pairs: int


def attitude_path(gyr: NDArray, bg: NDArray, dt: float, frame: str) -> NDArray[np.float64]:
    """A(t_k), k = 0..N, from rates (gyr_k - bg) held over each dt step: R(t_k) = R_0 A(t_k) for
    body rates and R(t_k) = A(t_k) R_0 for world rates."""
    out = np.empty((len(gyr) + 1, 3, 3))
    out[0] = np.eye(3)
    for k, w in enumerate(gyr):
        E = exp_so3((w - bg) * dt)
        out[k + 1] = out[k] @ E if frame == "body" else E @ out[k]
    return out


def relative_attitudes(gyr: NDArray, bg: NDArray, dt: float) -> NDArray[np.float64]:
    """Body-rate attitude path (kept for callers that only handle body rates)."""
    return attitude_path(gyr, bg, dt, "body")


def compose(A: NDArray, R0: NDArray, frame: str) -> NDArray[np.float64]:
    """Attitude R(t) from the path A(t) and the start attitude R_0."""
    return R0 @ A if frame == "body" else A @ R0


def wahba(w: NDArray, u: NDArray, sig: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R minimising sum ||w_i - R u_i||^2 / sig_i^2, and the covariance of its world-frame error
    (with the full-turn prior, see `_PRIOR_INFO`)."""
    B = sum(np.outer(wi, ui) / s**2 for wi, ui, s in zip(w, u, sig, strict=True))
    U, _, Vt = np.linalg.svd(B)
    D = np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(Vt)])
    R = U @ D @ Vt
    info = np.zeros((3, 3))
    for ui, s in zip(u, sig, strict=True):
        A = skew(R @ ui)
        info += A.T @ A / s**2
    return R, np.linalg.inv(info + _PRIOR_INFO)


# Prior on the start attitude: each angle uniform over a full turn (variance pi^2 / 3), so a
# direction the pairs do not determine (yaw when only gravity is seen) comes out as unknown.
_PRIOR_INFO = np.eye(3) * 3.0 / np.pi**2


def _fit(
    R0: NDArray,
    gyr: NDArray,
    acc: NDArray,
    w: NDArray,
    C: NDArray,
    dt: float,
    per: int,
    frame: str,
    bg_prior: NDArray,
    bg_sigma: NDArray,
    ba_prior: NDArray,
    ba_sigma: NDArray,
    outer: int = 3,
    inner: int = 10,
    vision: list[VisionPair] | None = None,
    R_bc: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Gauss-Newton over (start attitude, gyro bias, accel bias) on
    (w - pred)^T C^-1 (w - pred) over the stacked interval vectors (C: their full covariance)
    plus the bias priors. The gyro-bias
    effect is linearised as an attitude drift -G b_g t (G = I for world rates, R(t) for body
    rates) and the attitude path is recomputed with the new gyro bias between outer passes.
    Returns R_0, b_g, b_a, the 9x9 covariance and the attitude path A."""
    bg, ba = bg_prior.copy(), ba_prior.copy()
    n_int = len(w)
    t_mid = (np.arange(n_int * per) + 0.5) * dt
    info = np.eye(9)
    Ci = np.linalg.inv(C)
    for _ in range(outer):
        A = attitude_path(gyr, bg, dt, frame)
        half = np.array(
            [
                a @ exp_so3((g - bg) * dt / 2)
                if frame == "body"
                else exp_so3((g - bg) * dt / 2) @ a
                for a, g in zip(A[:-1], gyr, strict=True)
            ]
        )
        dbg_lin = np.zeros(3)  # gyro-bias change since the path was computed (linearised)
        for _ in range(inner):
            info = np.zeros((9, 9))
            info[:3, :3] = _PRIOR_INFO
            info[3:6, 3:6] = np.diag(1.0 / bg_sigma**2)
            info[6:9, 6:9] = np.diag(1.0 / ba_sigma**2)
            g = np.zeros(9)
            g[3:6] = (bg_prior - bg) / bg_sigma**2
            g[6:9] = (ba_prior - ba) / ba_sigma**2
            Js = np.zeros((3 * n_int, 9))
            rs = np.zeros(3 * n_int)
            for i in range(n_int):
                sl = slice(i * per, (i + 1) * per)
                Rk = np.array([compose(a, R0, frame) for a in half[sl]])
                fk = acc[sl] - ba
                Rf = np.einsum("nij,nj->ni", Rk, fk)
                Gk = np.broadcast_to(np.eye(3), Rk.shape) if frame == "world" else Rk
                S = np.array([skew(x) for x in Rf])
                drift = np.einsum("nij,njk->nik", S, Gk) * (t_mid[sl] * dt)[:, None, None]
                pred = Rf.sum(0) * dt + drift.sum(0) @ dbg_lin
                J = np.zeros((3, 9))
                if frame == "body":
                    J[:, :3] = -skew(Rf.sum(0) * dt)
                else:
                    J[:, :3] = -np.einsum("nij,njk->ik", S, half[sl]) * dt
                J[:, 3:6] = drift.sum(0)
                J[:, 6:9] = -Rk.sum(0) * dt
                Js[3 * i : 3 * i + 3] = J
                rs[3 * i : 3 * i + 3] = w[i] - pred
            info += Js.T @ Ci @ Js
            g += Js.T @ Ci @ rs
            # camera travel directions: R(t) R_bc d_c should be parallel to d_w; residual in
            # the plane perpendicular to d_w, per-axis variance (sigma_c^2 + sigma_w^2) / 2
            for vp in vision or []:
                kt = round(vp.t_s / dt)
                Rt = compose(A[kt], R0, frame)
                m = Rt @ R_bc @ vp.d_c
                Bt = _tangent_basis(vp.d_w)
                Gt = np.eye(3) if frame == "world" else Rt
                Jm = np.zeros((3, 9))
                Jm[:, :3] = -skew(m) @ (A[kt] if frame == "world" else np.eye(3))
                Jm[:, 3:6] = skew(m) @ Gt * vp.t_s
                Jv = Bt @ Jm
                rv = -(Bt @ m) - Bt @ (Jm[:, 3:6] @ dbg_lin)
                var = (vp.sigma_c**2 + vp.sigma_w**2) / 2.0
                info += Jv.T @ Jv / var
                g += Jv.T @ rv / var
            d = np.linalg.solve(info, g)
            R0 = exp_so3(d[:3]) @ R0
            bg = bg + d[3:6]
            dbg_lin = dbg_lin + d[3:6]
            ba = ba + d[6:9]
            if np.linalg.norm(d) < 1e-12:
                break
    return R0, bg, ba, np.linalg.inv(info), attitude_path(gyr, bg, dt, frame)


def _tangent_basis(d: NDArray) -> NDArray:
    """2x3 orthonormal basis of the plane perpendicular to unit vector d."""
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(d, e1)])


def align(
    gyr: ArrayLike,
    acc: ArrayLike,
    imu_dt: float,
    gps_vel: ArrayLike,
    gps_dt: float,
    bg_prior: ArrayLike,
    bg_sigma: ArrayLike,
    ba_prior: ArrayLike,
    ba_sigma: ArrayLike,
    acc_sigma: ArrayLike,
    vel_sigma: float,
    min_window_s: float,
    max_window_s: float,
    max_sigma_rad: float,
    gyro_frame: str = "body",
    gyr_sigma: ArrayLike = 0.0,
    vision: list[VisionPair] | None = None,
    R_bc: NDArray | None = None,
) -> Alignment:
    """Attitude and IMU biases from IMU step samples starting at t0 (each the mean over its
    step) and GPS velocities at t0, t0 + gps_dt, ...

    The window starts at `min_window_s` and grows one GPS interval at a time until the largest
    attitude standard deviation is below `max_sigma_rad` or `max_window_s` is reached (then the
    last fit is returned with its covariance, however large).
    """
    gyr = np.asarray(gyr, float)
    acc = np.asarray(acc, float)
    gv = np.asarray(gps_vel, float)
    bgp = np.asarray(bg_prior, float)
    bap = np.asarray(ba_prior, float)
    bgs = np.asarray(bg_sigma, float) * np.ones(3)
    bas = np.asarray(ba_sigma, float) * np.ones(3)
    per = round(gps_dt / imu_dt)
    n_int = min(len(gv) - 1, len(gyr) // per)
    # interval-vector noise: accelerometer white noise summed over the interval (independent
    # between intervals) and GPS velocity differences, which share a fix with their neighbours
    # (variance 2 s_v^2, covariance -s_v^2 between adjacent intervals)
    acc_var = np.asarray(acc_sigma, float) ** 2 * np.ones(3) * imu_dt**2 * per
    s_i = float(np.sqrt(2 * vel_sigma**2 + acc_var.max()))  # for the Wahba start only
    best = None
    for m in range(2, n_int + 1):
        win = m * gps_dt
        if win + 1e-9 < min_window_s:
            continue
        g_m, a_m = gyr[: m * per], acc[: m * per]
        w = np.array([gv[i + 1] - gv[i] - GRAVITY_NED * gps_dt for i in range(m)])
        # start: Wahba on (A_c^T w_i, sum f dt) (world) or (w_i, sum A f dt) (body), A at prior bg
        A = attitude_path(g_m, bgp, imu_dt, gyro_frame)
        pairs = []
        for i in range(m):
            sl = slice(i * per, (i + 1) * per)
            f = a_m[sl] - bap
            if gyro_frame == "body":
                pairs.append((w[i], np.einsum("nij,nj->i", A[:-1][sl], f) * imu_dt))
            else:
                pairs.append((A[(2 * i + 1) * per // 2].T @ w[i], f.sum(0) * imu_dt))
        sig = [s_i] * m
        vis = [vp for vp in vision or [] if vp.t_s + vp.dt_s <= win + 1e-9]
        for vp in vis:  # unit-vector pairs, angular sigma per axis
            At = A[round(vp.t_s / imu_dt)]
            if gyro_frame == "body":
                pairs.append((vp.d_w, At @ R_bc @ vp.d_c))
            else:
                pairs.append((At.T @ vp.d_w, R_bc @ vp.d_c))
            sig.append(float(np.sqrt((vp.sigma_c**2 + vp.sigma_w**2) / 2.0)))
        R_init, _ = wahba(
            np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]), np.array(sig)
        )
        D = np.eye(m + 1)[1:] - np.eye(m + 1)[:-1]  # w_i = v_i+1 - v_i
        C = np.kron(D @ D.T, np.eye(3) * vel_sigma**2) + np.kron(np.eye(m), np.diag(acc_var))
        # gyro white noise walks the attitude path: attitude error at interval i's centre has
        # covariance q t_i (q = s_g^2 dt per second, largest axis), shared up to min(t_i, t_j),
        # and moves interval vector i by -[w_i]x times it
        q = float(np.max(np.asarray(gyr_sigma, float) ** 2)) * imu_dt
        tc = (np.arange(m) + 0.5) * gps_dt
        S = [skew(wi) for wi in w]
        for i in range(m):
            for j in range(m):
                C[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += S[i] @ S[j].T * q * min(tc[i], tc[j])
        R0, bg, ba, cov, A = _fit(
            R_init,
            g_m,
            a_m,
            w,
            C,
            imu_dt,
            per,
            gyro_frame,
            bgp,
            bgs,
            bap,
            bas,
            vision=vis,
            R_bc=R_bc,
        )
        T = m * gps_dt
        R_end = compose(A[-1], R0, gyro_frame)
        Tm = np.eye(9)
        if gyro_frame == "world":  # A Exp(d) R0 = Exp(A d) A R0; bias drift -b_g T
            Tm[:3, :3] = A[-1]
            Tm[:3, 3:6] = -np.eye(3) * T
        else:  # Exp(d) R0 A; body-rate bias drift -R_end b_g T
            Tm[:3, 3:6] = -R_end * T
        best = Alignment(R0, bg, ba, cov, R_end, Tm @ cov @ Tm.T, win, m)
        if np.sqrt(np.max(np.linalg.eigvalsh(cov[:3, :3]))) <= max_sigma_rad or win >= max_window_s:
            break
    if best is None:
        raise ValueError("not enough IMU and GPS data for alignment")
    return best
