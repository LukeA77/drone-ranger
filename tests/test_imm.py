"""Interacting multiple model ground filter (stationary, constant velocity, coordinated turn).

Parameters here are test values chosen to make each behaviour visible, not project settings.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aerialrange.geometry.imm import (
    IMMGround,
    IMMSettings,
    ct_jacobian,
    ct_transition,
    gate_threshold,
)
from aerialrange.geometry.kalman import GroundKalman
from aerialrange.pipeline.filterbank import IMMFilterBank, TrackFilterBank, make_filter_bank

DT = 0.1
SET = IMMSettings(
    q_stationary=0.01,
    q_cv=0.5,
    q_ct=0.5,
    q_turn=0.05,
    turn_rate_sigma0=0.3,
    switch_rates=((0.0, 0.2, 0.1), (0.2, 0.0, 0.2), (0.1, 0.4, 0.0)),  # 1/s
    mode0=(1 / 3, 1 / 3, 1 / 3),
    velocity_sigma_moving=5.0,
    gate_prob=0.999,
    reset_after=3,
)


def run(truth: np.ndarray, sigma: float, seed: int = 0, settings: IMMSettings = SET):
    rng = np.random.default_rng(seed)
    z = truth + rng.normal(0, sigma, truth.shape)
    R = np.eye(2) * sigma**2
    f = IMMGround.from_two_points(z[0], R, z[1], R, DT, settings)
    out = []
    for k in range(2, len(z)):
        f.predict(DT)
        f.update(z[k], R)
        out.append((f.x.copy(), f.P.copy(), f.mode.copy()))
    return f, out


def circle(n: int, speed: float, rate: float) -> np.ndarray:
    t = np.arange(n) * DT
    r = speed / rate
    return np.column_stack([r * np.sin(rate * t), r * (1 - np.cos(rate * t))])


def test_ct_is_exact_on_a_circle():
    speed, rate = 8.0, 0.4
    x = np.array([0.0, 0.0, speed, 0.0, rate])
    truth = circle(2, speed, rate)
    nxt = ct_transition(x, DT)
    assert np.allclose(nxt[:2], truth[1], atol=1e-12)
    assert np.isclose(np.hypot(*nxt[2:4]), speed)
    assert nxt[4] == rate


def test_ct_tends_to_cv_as_turn_rate_vanishes():
    x = np.array([1.0, 2.0, 3.0, -4.0, 0.0])
    cv = np.array([1.0 + 3.0 * DT, 2.0 - 4.0 * DT, 3.0, -4.0, 0.0])
    assert np.allclose(ct_transition(x, DT), cv)
    assert np.allclose(ct_transition(x + [0, 0, 0, 0, 1e-9], DT), cv, atol=1e-8)


@pytest.mark.parametrize("w", [0.0, 1e-7, 0.3, -1.2])
def test_ct_jacobian_matches_finite_differences(w):
    x = np.array([5.0, -3.0, 6.0, 2.0, w])
    J = ct_jacobian(x, DT)
    eps = 1e-6
    num = np.column_stack(
        [(ct_transition(x + e, DT) - ct_transition(x - e, DT)) / (2 * eps) for e in np.eye(5) * eps]
    )
    assert np.allclose(J, num, atol=1e-6)


def test_gate_threshold_is_the_chi_square_quantile():
    # 2 degrees of freedom: P(chi2 <= t) = 1 - exp(-t / 2)
    assert np.isclose(gate_threshold(0.99), -2 * np.log(0.01))


def test_mode_probabilities_and_covariance_stay_valid():
    _, out = run(circle(200, 6.0, 0.3), 0.5)
    for _, P, mode in out:
        assert np.isclose(mode.sum(), 1.0) and (mode >= 0).all()
        assert np.allclose(P, P.T)
        assert np.linalg.eigvalsh(P).min() > -1e-9


def test_parked_object_settles_on_stationary_and_shows_little_speed():
    truth = np.zeros((300, 2)) + [40.0, -10.0]
    f, out = run(truth, 0.5)
    # A constant-velocity model whose velocity has settled near zero predicts almost the same,
    # so the two stay close; stationary must still be the most probable model.
    assert f.mode.argmax() == 0
    imm_speed = np.mean([np.hypot(*x[2:4]) for x, _, _ in out[100:]])
    rng = np.random.default_rng(0)
    z = truth + rng.normal(0, 0.5, truth.shape)
    kf = GroundKalman.from_two_points(z[0], np.eye(2) * 0.25, z[1], np.eye(2) * 0.25, DT, SET.q_cv)
    cv_speed = []
    for k in range(2, len(z)):
        kf.predict(DT)
        kf.update(z[k], np.eye(2) * 0.25)
        cv_speed.append(np.hypot(*kf.velocity))
    assert imm_speed < 0.5 * np.mean(cv_speed[98:])


def test_straight_driving_prefers_constant_velocity_and_gets_speed_right():
    t = np.arange(300) * DT
    truth = np.column_stack([10.0 * t, 2.0 * t])
    f, out = run(truth, 0.5)
    assert f.mode[1] > f.mode[0]
    speeds = np.array([np.hypot(*x[2:4]) for x, _, _ in out[100:]])
    assert abs(speeds.mean() - np.hypot(10, 2)) < 0.3


def test_turning_prefers_coordinated_turn_and_tracks_heading_better_than_cv():
    speed, rate = 8.0, 0.4
    truth = circle(400, speed, rate)
    f, out = run(truth, 0.3)
    assert f.mode[2] > f.mode[1]
    t = np.arange(2, 400) * DT
    true_head = rate * t
    imm_err = [
        abs(np.angle(np.exp(1j * (np.arctan2(x[3], x[2]) - h))))
        for (x, _, _), h in zip(out, true_head, strict=True)
    ]
    rng = np.random.default_rng(0)
    z = truth + rng.normal(0, 0.3, truth.shape)
    R = np.eye(2) * 0.09
    kf = GroundKalman.from_two_points(z[0], R, z[1], R, DT, SET.q_cv)
    cv_err = []
    for k, h in zip(range(2, len(z)), true_head, strict=True):
        kf.predict(DT)
        kf.update(z[k], R)
        cv_err.append(abs(np.angle(np.exp(1j * (np.arctan2(kf.x[3], kf.x[2]) - h)))))
    assert np.mean(imm_err[100:]) < np.mean(cv_err[100:])


def test_estimate_is_roughly_consistent_on_straight_motion():
    """Average NEES of position and velocity over Monte Carlo runs is near its expected value
    of 4. An IMM is an approximation, so the band is wider than for a single Kalman filter."""
    t = np.arange(150) * DT
    truth = np.column_stack([7.0 * t, -3.0 * t])
    vel = np.array([7.0, -3.0])
    nees = []
    for seed in range(40):
        _, out = run(truth, 0.5, seed)
        for k in range(60, len(out)):
            x, P, _ = out[k]
            e = np.concatenate([x[:2] - truth[k + 2], x[2:4] - vel])
            nees.append(e @ np.linalg.solve(P[:4, :4], e))
    assert 2.0 < np.mean(nees) < 8.0


def test_outlier_is_gated_out_and_a_run_of_them_resets_the_filter():
    t = np.arange(60) * DT
    truth = np.column_stack([5.0 * t, 0.0 * t])
    f, _ = run(truth, 0.2)
    R = np.eye(2) * 0.04
    f.predict(DT)
    before = f.x.copy()
    accepted = f.update(truth[-1] + [5.0 * DT + 50.0, 0.0], R)
    assert not accepted and f.rejections == 1
    assert np.allclose(f.x, before)  # a rejected measurement changes nothing
    far = truth[-1] + [80.0, 30.0]
    for _ in range(SET.reset_after - 1):
        f.predict(DT)
        f.update(far, R)
    assert f.needs_reset  # the caller restarts the track from two new measurements
    f.predict(DT)
    assert f.update(truth[-1] + [5.0 * DT * (SET.reset_after + 1), 0.0], R)
    assert f.rejections == 0 and not f.needs_reset  # an accepted measurement clears the run


def test_settings_reject_invalid_switch_rates():
    with pytest.raises(ValueError):
        replace(SET, switch_rates=((0.0, -0.2, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        replace(SET, switch_rates=((0.0, 0.1), (0.1, 0.0)))


def test_switch_matrix_is_stochastic_and_consistent_across_frame_rates():
    p1 = SET.switch_matrix(0.2)
    np.testing.assert_allclose(p1.sum(1), 1.0)
    np.testing.assert_allclose(SET.switch_matrix(0.1) @ SET.switch_matrix(0.1), p1, atol=1e-12)
    np.testing.assert_allclose(SET.switch_matrix(0.0), np.eye(3))
    # small dt: off-diagonal probability = rate x dt
    assert SET.switch_matrix(1e-4)[0, 1] == pytest.approx(0.2e-4, rel=1e-3)


# --- the per-track bank used by the pipeline -------------------------------------------------


IMM_CFG = {k: getattr(SET, k) for k in SET.__dataclass_fields__ if not k.startswith("_")}


def test_factory_keeps_version_1_unless_imm_is_named():
    km = {"q": 7.0, "max_unseen_frames": 30}
    assert isinstance(make_filter_bank(km, 10), TrackFilterBank)
    assert isinstance(
        make_filter_bank({**km, "motion_model": "imm", "imm": IMM_CFG}, 10), IMMFilterBank
    )
    with pytest.raises(ValueError):
        make_filter_bank({**km, "motion_model": "particle"}, 10)


def test_bank_outputs_match_version_1_keys_and_add_stationary_probability():
    bank = IMMFilterBank(SET, max_unseen=30)
    v1 = TrackFilterBank(q=1.0, max_unseen=30)
    for k in range(5):
        z = np.array([[float(k), 0.0], [50.0, 50.0]])
        out = bank.step(DT, [1, 2], z, [0.3, 0.3])
        ref = v1.step(DT, [1, 2], z, [0.3, 0.3])
    assert set(out) == set(ref) | {"p_stationary"}
    assert sorted(out["id"].tolist()) == [1, 2]
    p = dict(zip(out["id"].tolist(), out["p_stationary"], strict=True))
    assert p[2] > p[1]  # the parked object is the more likely to be stationary


def test_bank_restarts_a_track_after_a_jump_and_drops_unseen_tracks():
    bank = IMMFilterBank(SET, max_unseen=3)
    for k in range(20):
        bank.step(DT, [1], np.array([[0.5 * k, 0.0]]), [0.2])
    for _ in range(SET.reset_after):
        out = bank.step(DT, [1], np.array([[500.0, 500.0]]), [0.2])
    assert bank.resets == 1 and 1 not in out["id"].tolist()  # restarting from two points
    out = bank.step(DT, [1], np.array([[500.5, 500.0]]), [0.2])
    assert out["id"].tolist() == [1] and np.allclose(out["n_m"], 500.5, atol=0.3)
    for _ in range(3):
        out = bank.step(DT, [], np.zeros((0, 2)), np.zeros(0))
    assert out["id"].tolist() == []


def test_bank_takes_settings_per_group_and_full_covariances():
    slow = replace(SET, q_cv=0.01, q_ct=0.01)
    km = {"q": 7.0, "max_unseen_frames": 30, "motion_model": "imm"}
    per = {"person": IMM_CFG, "vehicle": {**IMM_CFG, "q_cv": 0.01, "q_ct": 0.01}}
    bank = make_filter_bank({**km, "imm": per}, 10)
    assert isinstance(bank, IMMFilterBank) and bank.settings_for("vehicle").q_cv == slow.q_cv
    R = np.array([[[0.09, 0.0], [0.0, 0.09]], [[4.0, 1.0], [1.0, 0.25]]])
    for k in range(3):
        z = np.array([[float(k), 0.0], [10.0 + k, 0.0]])
        out = bank.step(DT, [1, 2], z, R, ["person", "vehicle"])
    assert bank.filters[1].s.q_cv == SET.q_cv and bank.filters[2].s.q_cv == slow.q_cv
    bank.step(DT, [3], np.zeros((1, 2)), [0.1], ["animal"])  # first point: no filter yet
    with pytest.raises(KeyError):  # the filter starts on the second point
        bank.step(DT, [3], np.zeros((1, 2)), [0.1], ["animal"])
    # isotropic sigmas and the equivalent covariances give the same version 1 result
    a, b = TrackFilterBank(q=1.0, max_unseen=30), TrackFilterBank(q=1.0, max_unseen=30)
    for k in range(4):
        z = np.array([[float(k), 0.5 * k]])
        oa = a.step(DT, [1], z, [0.3])
        ob = b.step(DT, [1], z, (np.eye(2) * 0.09)[None])
    np.testing.assert_array_equal(oa["speed_ms"], ob["speed_ms"])
    assert len(out["id"]) == 2


def test_initial_modes_follow_the_two_point_velocity():
    from aerialrange.geometry.imm import initial_modes

    R = np.eye(2) * 0.01
    still = initial_modes([0.05, 0.0], R, R, 0.2, SET)
    fast = initial_modes([8.0, 0.0], R, R, 0.2, SET)
    assert still[0] > 0.9 and fast[0] < 1e-6
    np.testing.assert_allclose(fast[1] / fast[2], SET.mode0[1] / SET.mode0[2])
    # very noisy measurements: the velocity says little, so the prior is barely changed
    vague = initial_modes([1.0, 0.0], np.eye(2) * 100, np.eye(2) * 100, 0.2, SET)
    np.testing.assert_allclose(vague, SET.mode0, atol=0.05)


def test_a_bank_per_group_routes_measurements_by_group():
    from aerialrange.pipeline.filterbank import GroupedFilterBank

    km = {
        "q": 1.0,
        "max_unseen_frames": 30,
        "motion_model": {"person": "imm", "vehicle": "cv"},
        "imm": {"person": IMM_CFG},
    }
    bank = make_filter_bank(km, 10)
    assert isinstance(bank, GroupedFilterBank)
    assert isinstance(bank.banks["person"], IMMFilterBank)
    assert isinstance(bank.banks["vehicle"], TrackFilterBank)
    ref = TrackFilterBank(q=1.0, max_unseen=30)
    for k in range(4):
        z = np.array([[float(k), 0.0], [5.0 * k, 1.0]])
        out = bank.step(DT, [1, 2], z, [0.3, 0.3], ["person", "vehicle"])
        r = ref.step(DT, [2], z[1:], [0.3])
    got = dict(zip(out["id"].tolist(), out["speed_ms"], strict=True))
    assert set(got) == {1, 2} and got[2] == pytest.approx(r["speed_ms"][0])
    p = dict(zip(out["id"].tolist(), out["p_stationary"], strict=True))
    assert np.isnan(p[2]) and np.isfinite(p[1])
    with pytest.raises(ValueError):
        bank.step(DT, [1], np.zeros((1, 2)), [0.3])
