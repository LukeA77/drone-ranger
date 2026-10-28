"""Uncertainty propagation.

Analytic oracles for a nadir pinhole camera on a level drone, yaw 0:
image right = East, image down = South, so East X = (u - cx) h / fx and North = -(v - cy) h / fy.
"""

import numpy as np
import pytest

from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Camera, Pose
from aerialrange.geometry.ranging import RangeEstimate, estimate_range
from aerialrange.geometry.uncertainty import (
    PARAMS,
    build_covariance,
    ground_point_jacobian,
    nonlinearity_ratio,
    numerical_jacobian,
    propagate_uncertainty,
    reliability_flags,
)
from aerialrange.utils.seeds import make_rng

DEG = np.pi / 180
FX, FY, CX, CY = 800.0, 780.0, 640.0, 360.0  # synthetic
H = 20.0


def nadir():
    return Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(90 * DEG))


def level(p=(0.0, 0.0, -H), yaw=0.0):
    return Pose.from_ypr(p, yaw, 0.0, 0.0)


# --- numerical Jacobian vs analytic oracles --------------------------------------------------


@pytest.fixture
def jac():
    uv = (CX + 150.0, CY - 90.0)
    names, J, _ = ground_point_jacobian(nadir(), level(), uv, H, sigmas=dict.fromkeys(PARAMS, 1.0))
    return uv, dict(zip(names, J.T, strict=True))


def test_dX_dh_matches_oracle(jac):
    (u, _), cols = jac
    assert cols["h"][1] == pytest.approx((u - CX) / FX, rel=1e-6)


def test_dX_du_matches_oracle(jac):
    _, cols = jac
    assert cols["u"][1] == pytest.approx(H / FX, rel=1e-6)


def test_dX_dfx_matches_oracle(jac):
    (u, _), cols = jac
    assert cols["fx"][1] == pytest.approx(-(u - CX) * H / FX**2, rel=1e-6)


def test_dN_dv_matches_oracle(jac):
    _, cols = jac
    assert cols["v"][0] == pytest.approx(-H / FY, rel=1e-6)


def test_dX_dcx_matches_oracle(jac):
    _, cols = jac
    assert cols["cx"][1] == pytest.approx(-H / FX, rel=1e-6)


def test_numerical_jacobian_on_known_function():
    def f(x):
        return np.array([x[0] ** 2 * x[1], np.sin(x[1])])

    x = np.array([1.5, 0.3])
    J = numerical_jacobian(f, x, steps=np.array([1e-5, 1e-5]))
    np.testing.assert_allclose(J, [[2 * 1.5 * 0.3, 1.5**2], [0, np.cos(0.3)]], rtol=1e-7)


def test_zero_sigma_parameters_are_skipped():
    names, J, _ = ground_point_jacobian(nadir(), level(), (CX, CY), H, sigmas={"u": 1.0})
    assert names == ["u"] and J.shape == (3, 1)


# --- first-order propagation: closed-form checks ---------------------------------------------


def test_pixel_noise_only_gives_h_over_f_sigma():
    est = propagate_uncertainty(nadir(), level(), (CX + 100, CY), H, {"u": 2.0})
    assert np.sqrt(est.cov_ne[1, 1]) == pytest.approx(H / FX * 2.0, rel=1e-6)
    assert est.cov_ne[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_height_noise_on_oblique_view_gives_range_sigma_over_sin_depression():
    tilt = 35.0
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(tilt * DEG))
    est = propagate_uncertainty(cam, level(), (CX, CY), H, {"h": 0.5})
    assert est.range_sigma_m == pytest.approx(0.5 / np.sin(tilt * DEG), rel=1e-6)


def test_yaw_noise_maps_one_to_one_onto_bearing():
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(30 * DEG))
    est = propagate_uncertainty(cam, level(yaw=0.4), (CX, CY), H, {"yaw": 1.0 * DEG})
    assert est.azimuth_sigma_deg == pytest.approx(1.0, rel=1e-6)
    assert est.range_sigma_m == pytest.approx(0.0, abs=1e-9)


def test_tilt_noise_increases_depression_sensitivity():
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(30 * DEG))
    est = propagate_uncertainty(cam, level(), (CX, CY), H, {"tilt": 0.5 * DEG})
    # dR/d(delta) for R = h / sin(delta) is -h cos / sin^2.
    d = 30 * DEG
    assert est.range_sigma_m == pytest.approx(H * np.cos(d) / np.sin(d) ** 2 * 0.5 * DEG, rel=1e-5)


def test_obj_height_noise_only_matters_in_height_mode():
    sig = {"obj_height": 0.2}
    a = propagate_uncertainty(nadir(), level(), (CX + 200, CY), H, sig)
    b = propagate_uncertainty(
        nadir(), level(), (CX + 200, CY), H, sig, mode="centre_with_height", obj_height_m=1.5
    )
    assert np.allclose(a.cov_ne, 0.0)
    assert b.cov_ne[1, 1] > 0


def test_covariance_is_symmetric_psd_and_correlation_changes_it():
    names, S = build_covariance({"h": 0.5, "pitch": 0.01})
    assert names == ["h", "pitch"]
    np.testing.assert_allclose(S, np.diag([0.25, 1e-4]))
    _, S_c = build_covariance({"h": 0.5, "pitch": 0.01}, correlations={("h", "pitch"): 0.6})
    assert S_c[0, 1] == S_c[1, 0] == pytest.approx(0.6 * 0.5 * 0.01)
    assert np.all(np.linalg.eigvalsh(S_c) >= 0)
    with pytest.raises(ValueError):
        build_covariance({"h": 0.5}, correlations={("h", "pitch"): 0.5})
    with pytest.raises(ValueError):
        build_covariance({"h": 0.5, "pitch": 0.1}, correlations={("h", "pitch"): 1.5})
    with pytest.raises(ValueError):
        build_covariance({"altitude": 1.0})


def test_correlated_noise_changes_propagated_sigma():
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(30 * DEG))
    sig = {"h": 0.5, "pitch": 0.5 * DEG}
    a = propagate_uncertainty(cam, level(), (CX, CY), H, sig)
    b = propagate_uncertainty(cam, level(), (CX, CY), H, sig, correlations={("h", "pitch"): 0.8})
    assert a.range_sigma_m != pytest.approx(b.range_sigma_m)


def test_invalid_nominal_estimate_propagates_nothing():
    cam = Camera.pinhole(FX, FY, CX, CY)  # forward, principal point on horizon
    est = propagate_uncertainty(cam, level(), (CX, CY), H, {"u": 1.0})
    assert not est.valid
    assert np.isnan(est.range_sigma_m)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        propagate_uncertainty(nadir(), level(), (CX, CY), H, {"u": 1.0}, method="unscented")


def test_monte_carlo_requires_draw_count_and_rng():
    with pytest.raises(ValueError):
        propagate_uncertainty(nadir(), level(), (CX, CY), H, {"u": 1.0}, method="monte_carlo")


# --- first order vs Monte Carlo --------------------------------------------------------------

SIG_LINEAR = {"u": 1.0, "v": 1.0, "h": 0.3, "roll": 0.2 * DEG, "pitch": 0.2 * DEG}


def test_monte_carlo_agrees_with_first_order_in_near_linear_regime():
    args = (nadir(), level(), (CX + 120, CY + 80), H, SIG_LINEAR)
    fo = propagate_uncertainty(*args)
    mc = propagate_uncertainty(*args, method="monte_carlo", n_mc=20000, rng=make_rng(0))
    assert mc.mc is not None and mc.mc.n_valid == 20000
    # Tolerance: 5 per cent of the largest variance (well above the sampling error at 20,000).
    np.testing.assert_allclose(mc.cov_ne, fo.cov_ne, atol=0.05 * np.max(np.diag(fo.cov_ne)))
    assert nonlinearity_ratio(fo, mc) == pytest.approx(1.0, abs=0.05)


def test_monte_carlo_is_reproducible_with_seed():
    args = (nadir(), level(), (CX + 120, CY), H, SIG_LINEAR)
    a = propagate_uncertainty(*args, method="monte_carlo", n_mc=500, rng=make_rng(5))
    b = propagate_uncertainty(*args, method="monte_carlo", n_mc=500, rng=make_rng(5))
    np.testing.assert_array_equal(a.cov_ne, b.cov_ne)


def test_near_horizon_monte_carlo_is_skewed_and_first_order_underestimates():
    """Shallow ray (about 3 degrees depression): range is a 1/sin function, so it is skewed."""
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(3 * DEG))
    sig = {"pitch": 0.5 * DEG, "tilt": 0.5 * DEG}
    fo = propagate_uncertainty(cam, level(), (CX, CY), H, sig)
    mc = propagate_uncertainty(
        cam, level(), (CX, CY), H, sig, method="monte_carlo", n_mc=20000, rng=make_rng(1)
    )
    p = mc.mc.range_percentiles
    upper, lower = p[97.5] - p[50.0], p[50.0] - p[2.5]
    assert upper > 1.5 * lower  # strongly right-skewed
    assert nonlinearity_ratio(fo, mc) > 1.2


def test_monte_carlo_counts_invalid_draws_near_horizon():
    cam = Camera.pinhole(FX, FY, CX, CY, R_bc=fr.R_bc_from_tilt(1 * DEG))
    mc = propagate_uncertainty(
        cam,
        level(),
        (CX, CY),
        H,
        {"tilt": 1.0 * DEG},
        method="monte_carlo",
        n_mc=2000,
        rng=make_rng(2),
    )
    assert 0 < mc.mc.n_valid < 2000
    assert "mc_invalid_draws" in mc.flags


# --- reliability flag ------------------------------------------------------------------------


def _est(**kw):
    base = dict(
        point_w=np.zeros(3),
        range_m=100.0,
        ground_dist_m=90.0,
        azimuth_deg=0.0,
        rel_azimuth_deg=0.0,
        depression_deg=30.0,
        valid=True,
        reason="ok",
        range_sigma_m=5.0,
    )
    base.update(kw)
    return RangeEstimate(**base)


def test_reliability_flags():
    assert reliability_flags(_est(), rel_unc_threshold=0.1, min_depression_deg=5.0) == ()
    assert "invalid" in reliability_flags(_est(valid=False, reason="x"), 0.1, 5.0)
    assert "high_relative_uncertainty" in reliability_flags(_est(range_sigma_m=20.0), 0.1, 5.0)
    assert "below_min_depression" in reliability_flags(_est(depression_deg=2.0), 0.1, 5.0)
    assert "no_uncertainty" in reliability_flags(_est(range_sigma_m=float("nan")), 0.1, 5.0)


def test_estimate_without_uncertainty_has_nan_sigma():
    est = estimate_range(nadir(), level(), (CX, CY), H)
    assert np.isnan(est.range_sigma_m)


def test_point_directly_below_has_undefined_bearing_sigma():
    est = propagate_uncertainty(nadir(), level(), (CX, CY), H, {"u": 1.0, "v": 1.0})
    assert np.isnan(est.azimuth_sigma_deg)
    assert est.ground_dist_sigma_m == pytest.approx(np.hypot(H / FX, H / FY), rel=1e-6)
