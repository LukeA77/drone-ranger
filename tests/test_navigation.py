"""Tests for the Phase 11 filter (D-741): rotations, IMU noise estimate, propagation,
Jacobians, alignment and consistency. All data here is synthetic."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from aerialrange.navigation.alignment import GRAVITY_NED, align, wahba
from aerialrange.navigation.gnss import GnssErrorModel
from aerialrange.navigation.imu import ImuNoise, white_noise_sigma
from aerialrange.navigation.msckf import (
    BA,
    BG,
    GC,
    GM,
    NCORE,
    PO,
    TH,
    VE,
    Msckf,
    MsckfSettings,
    camera_projection,
)
from aerialrange.navigation.so3 import exp_so3, log_so3, skew

DT = 0.01


def truth(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth synthetic flight: position, velocity, acceleration (NED) and attitude R_wb."""
    p = np.c_[10 * t + 20 * np.sin(0.2 * t), 15 * np.sin(0.15 * t), -30 - 2 * np.sin(0.3 * t)]
    v = np.c_[10 + 4 * np.cos(0.2 * t), 2.25 * np.cos(0.15 * t), -0.6 * np.cos(0.3 * t)]
    a = np.c_[-0.8 * np.sin(0.2 * t), -0.3375 * np.sin(0.15 * t), 0.18 * np.sin(0.3 * t)]
    R = np.array(
        [exp_so3([0.1 * np.sin(0.5 * x), 0.05 * np.cos(0.3 * x), 0.3 * np.sin(0.1 * x)]) for x in t]
    )
    return p, v, a, R


def ideal_imu(t: np.ndarray, frame: str = "body") -> tuple[np.ndarray, np.ndarray]:
    """Noise-free gyro and accelerometer step samples consistent with `truth` (held for DT);
    gyro rates in the body or the world frame."""
    _, _, a, R = truth(np.r_[t, t[-1] + DT])
    if frame == "body":
        gyr = np.array([log_so3(R[k].T @ R[k + 1]) / DT for k in range(len(t))])
        R_mid = np.array([R[k] @ exp_so3(gyr[k] * DT / 2) for k in range(len(t))])
    else:
        gyr = np.array([log_so3(R[k + 1] @ R[k].T) / DT for k in range(len(t))])
        R_mid = np.array([exp_so3(gyr[k] * DT / 2) @ R[k] for k in range(len(t))])
    _, _, a_mid, _ = truth(t + DT / 2)
    acc = np.einsum("nji,nj->ni", R_mid, a_mid - GRAVITY_NED)
    return gyr, acc


def gnss_model() -> GnssErrorModel:
    return GnssErrorModel(np.full(3, 0.8), np.full(3, 0.5), 5.0, np.full(3, 0.5), 0.074)


def make_filter(R, p, v, P0, noise, fej=True, frame="body") -> Msckf:
    return Msckf(
        np.eye(3),
        np.zeros(3),
        500.0,
        MsckfSettings(fej=fej, gyro_frame=frame),
        noise,
        gnss_model(),
        0.0,
        R,
        p,
        v,
        np.zeros(3),
        np.zeros(3),
        P0,
    )


def tiny_noise() -> ImuNoise:
    z = np.full(3, 1e-9)
    return ImuNoise(z, z, z, z)


def test_so3_round_trip_and_skew() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.normal(size=3) * rng.uniform(0, 3)
        R = exp_so3(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-10)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))
    assert np.allclose(log_so3(exp_so3([0, 0, np.pi - 1e-7])), [0, 0, np.pi - 1e-7], atol=1e-5)


def test_white_noise_sigma_ignores_smooth_motion() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(20000) * DT
    smooth = np.c_[np.sin(0.5 * t), 3 * np.cos(0.2 * t), t * 0.1]
    sig = np.array([0.002, 0.03, 0.1])
    est = white_noise_sigma(smooth + rng.normal(size=smooth.shape) * sig)
    assert np.allclose(est / sig, 1.0, atol=0.03)


@pytest.mark.parametrize("frame", ["body", "world"])
def test_propagation_follows_continuous_truth(frame: str) -> None:
    t = np.arange(0, 20, DT)
    gyr, acc = ideal_imu(t, frame)
    p, v, _, R = truth(t)
    f = make_filter(R[0], p[0], v[0], np.eye(NCORE) * 1e-6, tiny_noise(), frame=frame)
    f.propagate(gyr[:-1], acc[:-1], DT)
    assert np.linalg.norm(f.p - p[-1]) < 0.05
    assert np.linalg.norm(f.v - v[-1]) < 0.01
    assert np.linalg.norm(log_so3(f.R.T @ R[-1])) < 1e-6


@pytest.mark.parametrize("frame", ["body", "world"])
def test_propagation_jacobian_matches_finite_differences(frame: str) -> None:
    t = np.arange(0, 0.5, DT)
    gyr, acc = ideal_imu(t, frame)
    p, v, _, R = truth(t)
    base = make_filter(R[0], p[0], v[0], np.eye(NCORE) * 1e-6, tiny_noise(), fej=False, frame=frame)
    base.propagate(gyr, acc, DT)
    Phi = base.last_phi
    eps = 1e-6
    for i in range(15):  # attitude, position, velocity, gyro bias, accel bias
        dx = np.zeros(NCORE)
        dx[i] = eps
        f = make_filter(
            R[0], p[0], v[0], np.eye(NCORE) * 1e-6, tiny_noise(), fej=False, frame=frame
        )
        f._apply(dx)
        f.propagate(gyr, acc, DT)
        err = np.r_[
            log_so3(f.R @ base.R.T), f.p - base.p, f.v - base.v, f.bg - base.bg, f.ba - base.ba
        ]
        assert np.allclose(err / eps, Phi[:15, i], atol=2e-3 * max(1.0, np.abs(Phi[:15, i]).max()))


def test_camera_projection_jacobians() -> None:
    rng = np.random.default_rng(2)
    R_bc = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t_bc = np.array([0.1, -0.05, 0.02])
    R = exp_so3(rng.normal(size=3) * 0.2)
    p = rng.normal(size=3)
    X = p + R @ (R_bc @ np.array([1.0, -2.0, 30.0]) + t_bc)
    z, Jth, Jp, Jx = camera_projection(R, p, R_bc, t_bc, X)
    eps = 1e-6
    for i in range(3):
        e = np.eye(3)[i] * eps
        assert np.allclose(
            (camera_projection(exp_so3(e) @ R, p, R_bc, t_bc, X)[0] - z) / eps, Jth[:, i], atol=1e-5
        )
        assert np.allclose(
            (camera_projection(R, p + e, R_bc, t_bc, X)[0] - z) / eps, Jp[:, i], atol=1e-5
        )
        assert np.allclose(
            (camera_projection(R, p, R_bc, t_bc, X + e)[0] - z) / eps, Jx[:, i], atol=1e-5
        )


def test_wahba_recovers_rotation_and_flags_unobservable_yaw() -> None:
    rng = np.random.default_rng(3)
    R0 = exp_so3([0.1, -0.2, 1.3])
    u = rng.normal(size=(4, 3)) * 3
    R, cov = wahba(u @ R0.T, u, np.full(4, 0.01))
    assert np.linalg.norm(log_so3(R.T @ R0)) < 1e-9
    assert np.sqrt(cov.diagonal().max()) < 0.01
    up = np.array([[0, 0, -9.81], [0, 0, -9.81]])  # only gravity: yaw unobservable
    _, cov2 = wahba(up @ R0.T + 1e-9, up, np.full(2, 0.01))
    assert np.sqrt(np.linalg.eigvalsh(cov2).max()) > 1.0


@pytest.mark.parametrize("frame", ["body", "world"])
def test_alignment_on_noise_free_flight(frame: str) -> None:
    t = np.arange(0, 6, DT)
    gyr, acc = ideal_imu(t, frame)
    _, v, _, R = truth(np.arange(0, 7))
    a = align(
        gyr,
        acc,
        DT,
        v,
        1.0,
        np.zeros(3),
        np.full(3, 1e-5),
        np.zeros(3),
        np.full(3, 1e-4),
        np.full(3, 1e-4),
        1e-3,
        2.0,
        6.0,
        np.deg2rad(5),
        frame,
    )
    assert np.linalg.norm(log_so3(a.R0.T @ R[0])) < np.deg2rad(0.1)


def test_alignment_covariance_is_honest_with_biases_and_noise() -> None:
    """Monte Carlo: biases drawn from their priors, IMU and GPS-velocity noise; the NEES of
    (start attitude, gyro bias, accel bias) averaged over runs lies in the 95 % band."""
    runs = 25
    t = np.arange(0, 8, DT)
    gyr0, acc0 = ideal_imu(t, "world")
    _, v, _, R = truth(np.arange(0, 9))
    bg_s, ba_s = np.full(3, 2e-3), np.full(3, 0.05)
    sg, sa, sv = 0.01, 0.05, 0.074
    nees = []
    for run in range(runs):
        rng = np.random.default_rng(500 + run)
        bg = rng.normal(size=3) * bg_s
        ba = rng.normal(size=3) * ba_s
        gyr = gyr0 + bg + rng.normal(size=gyr0.shape) * sg
        acc = acc0 + ba + rng.normal(size=acc0.shape) * sa
        a = align(
            gyr,
            acc,
            DT,
            v + rng.normal(size=v.shape) * sv,
            1.0,
            np.zeros(3),
            bg_s,
            np.zeros(3),
            ba_s,
            np.full(3, sa),
            sv,
            8.0,
            8.0,
            0.0,
            "world",
            np.full(3, sg),
        )
        e = np.r_[log_so3(R[0] @ a.R0.T), bg - a.bg, ba - a.ba]
        nees.append(e @ np.linalg.solve(a.cov, e))
    lo, hi = chi2.ppf([0.025, 0.975], 9 * runs) / runs
    assert lo <= np.mean(nees) <= hi, (np.mean(nees), lo, hi)


def test_filter_consistent_with_imu_and_gps_only() -> None:
    """Monte Carlo NEES of (attitude, position, velocity) with IMU and GPS (no camera): the mean
    over runs and time must lie within the two-sided 95 % chi-squared band."""
    runs, dur = 12, 20.0
    t = np.arange(0, dur, DT)
    gyr0, acc0 = ideal_imu(t)
    p, v, _, R = truth(t)
    sig_g, sig_a = np.array([0.01, 0.02, 0.005]), np.array([0.05, 0.02, 0.1])
    noise = ImuNoise(sig_g, sig_a, np.full(3, 1e-5), np.full(3, 2e-4))
    g = gnss_model()
    nees = []
    for run in range(runs):
        rng = np.random.default_rng(100 + run)
        bg = rng.normal(size=3) * 1e-3
        ba = rng.normal(size=3) * 1e-2
        bgs = bg + np.cumsum(rng.normal(size=(len(t), 3)) * noise.gyr_bias_rw, axis=0)
        bas = ba + np.cumsum(rng.normal(size=(len(t), 3)) * noise.acc_bias_rw, axis=0)
        gyr = gyr0 + bgs + rng.normal(size=gyr0.shape) * sig_g
        acc = acc0 + bas + rng.normal(size=acc0.shape) * sig_a
        c = rng.normal(size=3) * g.const_sigma
        m = rng.normal(size=3) * g.gm_sigma
        P0 = np.zeros((NCORE, NCORE))
        P0[TH, TH] = np.eye(3) * np.deg2rad(1) ** 2
        P0[PO, PO] = np.diag(g.const_sigma**2 + g.gm_sigma**2 + g.white_sigma**2)
        P0[PO, GC] = P0[GC, PO] = -np.diag(g.const_sigma**2)
        P0[PO, GM] = P0[GM, PO] = -np.diag(g.gm_sigma**2)
        P0[GC, GC] = np.diag(g.const_sigma**2)
        P0[GM, GM] = np.diag(g.gm_sigma**2)
        P0[VE, VE] = np.eye(3) * g.vel_sigma**2
        P0[BG, BG] = np.eye(3) * 1e-6
        P0[BA, BA] = np.eye(3) * 1e-4
        th0 = rng.normal(size=3) * np.deg2rad(1)
        z0 = p[0] + c + m + rng.normal(size=3) * g.white_sigma
        f = make_filter(
            exp_so3(-th0) @ R[0], z0, v[0] + rng.normal(size=3) * g.vel_sigma, P0, noise
        )
        a_gm = np.exp(-1.0 / g.gm_tau_s)
        per = round(1.0 / DT)
        for s in range(1, int(dur)):
            f.propagate(gyr[(s - 1) * per : s * per], acc[(s - 1) * per : s * per], DT)
            m = a_gm * m + rng.normal(size=3) * g.gm_sigma * np.sqrt(1 - a_gm**2)
            k = s * per
            f.update_gps(
                p[k] + c + m + rng.normal(size=3) * g.white_sigma,
                v[k] + rng.normal(size=3) * g.vel_sigma,
            )
            e = np.r_[log_so3(R[k] @ f.R.T), p[k] - f.p, v[k] - f.v]
            nees.append(e @ np.linalg.solve(f.P[:9, :9], e))
    # errors within a run are correlated in time, so only runs count as independent samples
    lo, hi = chi2.ppf([0.025, 0.975], 9 * runs) / runs
    assert lo <= np.mean(nees) <= hi, (np.mean(nees), lo, hi)


def test_camera_travel_direction_fixes_yaw_in_steady_flight() -> None:
    """Straight, constant-velocity flight: IMU and GPS see only gravity, so yaw is unknown;
    camera travel directions against GPS directions recover it."""
    from aerialrange.navigation.alignment import VisionPair

    R_bc = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Rwb = exp_so3([0.02, -0.05, 1.0])
    n = 600
    gyr = np.zeros((n, 3))
    acc = np.tile(Rwb.T @ (-GRAVITY_NED), (n, 1))
    vel = np.tile([8.0, 3.0, -0.5], (7, 1))
    args = (
        np.zeros(3),
        np.full(3, 1e-5),
        np.zeros(3),
        np.full(3, 1e-3),
        np.full(3, 1e-3),
        0.074,
        6.0,
        6.0,
        0.0,
        "world",
        np.full(3, 1e-4),
    )
    a0 = align(gyr, acc, DT, vel, 1.0, *args)
    assert np.sqrt(a0.cov[2, 2]) > 0.5  # yaw unknown without the camera
    rng = np.random.default_rng(7)
    dw = vel[0] / np.linalg.norm(vel[0])
    pairs = []
    for i in range(5):
        dc = R_bc.T @ Rwb.T @ dw + rng.normal(size=3) * 0.02
        pairs.append(VisionPair(float(i), 0.4, dc / np.linalg.norm(dc), 0.03, dw, 0.01))
    a1 = align(gyr, acc, DT, vel, 1.0, *args, vision=pairs, R_bc=R_bc)
    yaw_err = log_so3(Rwb @ a1.R0.T)[2]
    assert np.sqrt(a1.cov[2, 2]) < 0.03
    assert abs(yaw_err) < 3 * np.sqrt(a1.cov[2, 2])
