"""Ground-plane constant-velocity Kalman filter."""

import numpy as np
import pytest
from scipy.stats import chi2

from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Camera, Pose, project_world_point
from aerialrange.geometry.kalman import GroundKalman, cv_process_noise, cv_transition, run_filter
from aerialrange.geometry.uncertainty import propagate_uncertainty
from aerialrange.utils.seeds import make_rng

# Synthetic simulation settings (test fixtures, not tuned parameters).
DT = 0.2
Q = 0.05  # m^2 / s^3
N_STEPS = 60
N_RUNS = 200
CONF = 0.99


def simulate(rng, x0, q, n=N_STEPS, dt=DT):
    """Truth from the same CV model the filter assumes; heteroscedastic measurement noise."""
    xs, zs, Rs = [], [], []
    x = np.asarray(x0, float)
    L = np.linalg.cholesky(cv_process_noise(dt, q)) if q > 0 else np.zeros((4, 4))
    for k in range(n):
        if k:
            x = cv_transition(dt) @ x + L @ rng.normal(size=4)
        s = rng.uniform(0.3, 2.0, size=2)
        R = np.diag(s**2)
        xs.append(x.copy())
        zs.append(x[:2] + s * rng.normal(size=2))
        Rs.append(R)
    return np.array(xs), zs, Rs


def test_transition_and_process_noise_shapes():
    F_ = cv_transition(0.5)
    np.testing.assert_allclose(F_ @ [1, 2, 3, 4], [2.5, 4.0, 3, 4])
    Qm = cv_process_noise(0.5, 2.0)
    np.testing.assert_allclose(Qm, Qm.T)
    assert Qm[0, 0] == pytest.approx(2.0 * 0.5**3 / 3)
    assert Qm[0, 2] == pytest.approx(2.0 * 0.5**2 / 2)
    assert Qm[2, 2] == pytest.approx(2.0 * 0.5)
    assert np.all(np.linalg.eigvalsh(Qm) >= -1e-15)


def test_negative_dt_or_q_rejected():
    with pytest.raises(ValueError):
        cv_transition(-0.1)
    with pytest.raises(ValueError):
        cv_process_noise(0.1, -1.0)


def test_two_point_initialisation():
    R = np.eye(2) * 0.25
    kf = GroundKalman.from_two_points([0, 0], R, [1.0, 2.0], R, dt=0.5, q=Q)
    np.testing.assert_allclose(kf.x, [1.0, 2.0, 2.0, 4.0])
    np.testing.assert_allclose(kf.P[2:, 2:], (R + R) / 0.25)
    np.testing.assert_allclose(kf.P[:2, 2:], R / 0.5)


def test_predict_only_follows_constant_velocity_and_grows_covariance():
    kf = GroundKalman(x=np.array([0, 0, 1.0, -2.0]), P=np.eye(4), q=Q)
    tr0 = np.trace(kf.P)
    kf.predict(1.0)
    np.testing.assert_allclose(kf.x, [1.0, -2.0, 1.0, -2.0])
    assert np.trace(kf.P) > tr0


def test_update_with_precise_measurement_pulls_to_it():
    kf = GroundKalman(x=np.zeros(4), P=np.eye(4) * 100, q=Q)
    kf.update([5.0, -3.0], np.eye(2) * 1e-6)
    np.testing.assert_allclose(kf.position, [5.0, -3.0], atol=1e-4)


def test_speed_and_heading_convention():
    kf = GroundKalman(x=np.array([0, 0, 0.0, 3.0]), P=np.eye(4) * 0.01, q=Q)
    speed, s_sig, heading, h_sig = kf.speed_heading()
    assert speed == pytest.approx(3.0)
    assert heading == pytest.approx(90.0)  # moving East
    assert s_sig == pytest.approx(0.1)
    assert h_sig == pytest.approx(np.degrees(0.1 / 3.0))
    kf.x[2:] = 0
    _, _, heading, h_sig = kf.speed_heading()
    assert np.isnan(heading) and np.isnan(h_sig)


def test_nees_is_consistent_for_moving_target():
    """Average NEES over runs lies inside the two-sided chi-squared interval (4 dof)."""
    rng = make_rng(0)
    nees = np.zeros((N_RUNS, N_STEPS - 1))
    for r in range(N_RUNS):
        xs, zs, Rs = simulate(rng, [0, 0, 2.0, -1.0], Q)
        states = run_filter([DT] * (N_STEPS - 1), zs, Rs, Q)
        for k, kf_x_P in enumerate(states):
            x, P = kf_x_P
            e = x - xs[k + 1]
            nees[r, k] = e @ np.linalg.solve(P, e)
    lo, hi = chi2.ppf([(1 - CONF) / 2, (1 + CONF) / 2], 4 * N_RUNS) / N_RUNS
    mean_nees = nees.mean(axis=0)
    inside = np.mean((mean_nees > lo) & (mean_nees < hi))
    assert inside >= CONF - 0.03, (inside, lo, hi, mean_nees.min(), mean_nees.max())
    # The time-averaged NEES should be close to the state dimension.
    assert mean_nees.mean() == pytest.approx(4.0, rel=0.1)


def test_stationary_target_speed_consistent_with_zero():
    """Normalised velocity error (2 dof) against true zero velocity stays in its interval."""
    rng = make_rng(1)
    vals = []
    for _ in range(N_RUNS):
        _, zs, Rs = simulate(rng, [10.0, -5.0, 0.0, 0.0], q=0.0)
        x, P = run_filter([DT] * (N_STEPS - 1), zs, Rs, Q)[-1]
        v = x[2:]
        vals.append(v @ np.linalg.solve(P[2:, 2:], v))
    lo, hi = chi2.ppf([(1 - CONF) / 2, (1 + CONF) / 2], 2 * N_RUNS) / N_RUNS
    # q > 0 in the filter but 0 in truth, so the filter is conservative: mean <= 2.
    assert np.mean(vals) < hi


def test_missed_detections_are_predict_only():
    rng = make_rng(2)
    _, zs, Rs = simulate(rng, [0, 0, 1.0, 1.0], Q, n=10)
    zs[5] = None
    Rs[5] = None
    states = run_filter([DT] * 9, zs, Rs, Q)
    assert len(states) == 9
    P_before, P_gap = states[3][1], states[4][1]  # index 4 = measurement 5 (missed)
    assert np.trace(P_gap) > np.trace(P_before)


def test_run_filter_validates_lengths():
    with pytest.raises(ValueError):
        run_filter([DT], [np.zeros(2)], [np.eye(2)], Q)


def test_world_frame_filter_sees_no_motion_when_only_the_drone_moves():
    """Drone flies past a stationary ground point; ranged world points feed the filter.

    The drone's own motion must not appear as target velocity.
    """
    rng = make_rng(3)
    cam = Camera.pinhole(800.0, 800.0, 640.0, 360.0, R_bc=fr.R_bc_from_tilt(60 * np.pi / 180))
    target = np.array([30.0, 5.0, 0.0])
    sig = {"u": 1.0, "v": 1.0}
    vals = []
    for _ in range(50):
        zs, Rs = [], []
        for k in range(N_STEPS):
            pose = Pose.from_ypr([k * DT * 3.0, 0.0, -20.0], 0.0, 0.0, 0.0)  # 3 m/s north
            uv, _ = project_world_point(cam, pose, target)
            uv = uv + rng.normal(size=2) * 1.0
            est = propagate_uncertainty(cam, pose, uv, 20.0, sig)
            zs.append(est.point_w[:2])
            Rs.append(est.cov_ne)
        x, P = run_filter([DT] * (N_STEPS - 1), zs, Rs, Q)[-1]
        v = x[2:]
        vals.append(v @ np.linalg.solve(P[2:, 2:], v))
    assert np.mean(vals) < chi2.ppf(0.99, 2 * 50) / 50


def test_two_point_initialisation_needs_positive_dt():
    with pytest.raises(ValueError):
        GroundKalman.from_two_points([0, 0], np.eye(2), [1, 1], np.eye(2), dt=0.0, q=Q)


def test_run_filter_needs_first_two_measurements():
    with pytest.raises(ValueError):
        run_filter([DT, DT], [None, np.zeros(2), np.zeros(2)], [None, np.eye(2), np.eye(2)], Q)


def test_zero_speed_reports_isotropic_speed_sigma():
    kf = GroundKalman(x=np.zeros(4), P=np.eye(4) * 0.04, q=Q)
    speed, s_sig, _, _ = kf.speed_heading()
    assert speed == 0.0 and s_sig == pytest.approx(0.2)


def test_velocity_property():
    kf = GroundKalman(x=np.array([1.0, 2.0, 3.0, 4.0]), P=np.eye(4), q=Q)
    np.testing.assert_array_equal(kf.velocity, [3.0, 4.0])


def test_batch_filter_matches_the_scalar_filter_exactly():
    from aerialrange.geometry.kalman import GroundKalman, GroundKalmanBatch

    rng = np.random.default_rng(3)
    n = 7
    z1, z2 = rng.normal(0, 5, (n, 2)), rng.normal(0, 5, (n, 2))
    R = np.stack([np.diag(rng.uniform(0.1, 2.0, 2)) for _ in range(n)])
    single = [GroundKalman.from_two_points(z1[i], R[i], z2[i], R[i], 0.2, 7.0) for i in range(n)]
    batch = GroundKalmanBatch.from_two_points(z1, R, z2, R, 0.2, 7.0)
    for step in range(6):
        dt = 0.2 * (1 + step % 2)
        z = rng.normal(0, 5, (n, 2))
        have = rng.random(n) > 0.3  # some objects unmeasured this frame: predict only
        for i, kf in enumerate(single):
            kf.predict(dt)
            if have[i]:
                kf.update(z[i], R[i])
        batch.predict(dt)
        batch.update(z, R, have)
        np.testing.assert_allclose(batch.x, [kf.x for kf in single], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batch.P, [kf.P for kf in single], rtol=1e-12, atol=1e-12)
    sp, sp_sig = batch.speed()
    for i, kf in enumerate(single):
        s, s_sig, _, _ = kf.speed_heading()
        assert sp[i] == pytest.approx(s) and sp_sig[i] == pytest.approx(s_sig)
