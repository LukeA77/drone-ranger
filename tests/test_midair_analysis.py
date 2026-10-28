"""Analysis from cached samples, on SYNTHETIC caches with known geometry."""

import numpy as np
import pytest

from aerialrange.data.midair import midair_camera
from aerialrange.evaluate import midair_analysis as ana
from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Pose, pixel_to_world_ray

DEG = np.pi / 180
W = 1024


def synthetic_cache(n_frames=40, n_pix=300, plane_n=(0.0, 0.0, -1.0), seed=0):
    """Samples on the plane n . X = 0 (through the origin) seen by a tilted-down drone."""
    rng = np.random.default_rng(seed)
    cam = midair_camera(W, W)
    n_w = np.asarray(plane_n, float)
    n_w = n_w / np.linalg.norm(n_w)
    cols = {
        k: []
        for k in (
            "group",
            "frame",
            "u",
            "v",
            "img_w",
            "img_h",
            "pose_p",
            "pose_q",
            "true_point_w",
            "true_range_m",
            "normal_w",
            "z_oracle",
            "z_rel_ref",
            "h_oracle_m",
        )
    }
    for k in range(n_frames):
        pose = Pose.from_ypr(
            [rng.uniform(-5, 5), rng.uniform(-5, 5), -rng.uniform(15, 40)],
            rng.uniform(-np.pi, np.pi),
            -rng.uniform(20, 50) * DEG,
            rng.uniform(-5, 5) * DEG,
        )
        uv = np.stack([rng.uniform(50, W - 50, n_pix), rng.uniform(600, W - 50, n_pix)], 1)
        C, d = pixel_to_world_ray(cam, pose, uv)
        lam = -(n_w @ C) / (d @ n_w)
        keep = lam > 0
        uv, X = uv[keep], C + lam[keep, None] * d[keep]
        m = len(uv)
        z_med = float(np.median(X[:, 2]))
        for key, val in (
            ("group", np.full(m, seed)),
            ("frame", np.full(m, k)),
            ("u", uv[:, 0]),
            ("v", uv[:, 1]),
            ("img_w", np.full(m, W)),
            ("img_h", np.full(m, W)),
            ("pose_p", np.tile(pose.p_w, (m, 1))),
            ("pose_q", np.tile(fr.R_to_quat(pose.R_wb), (m, 1))),
            ("true_point_w", X),
            ("true_range_m", np.linalg.norm(X - C, axis=1)),
            ("normal_w", np.tile(n_w, (m, 1))),
            ("z_oracle", np.full(m, z_med)),
            ("z_rel_ref", np.full(m, z_med)),
            ("h_oracle_m", np.full(m, z_med - C[2])),
        ):
            cols[key].append(val)
    return {k: np.concatenate(v) for k, v in cols.items()}


SIG = {
    "u": 1.0,
    "v": 1.0,
    "h": 0.3,
    "roll": 0.3 * DEG,
    "pitch": 0.3 * DEG,
    "yaw": 0.5 * DEG,
    "tilt": 0.2 * DEG,
    "fx": 2.0,
    "fy": 2.0,
}


def test_zero_noise_reproduces_noise_free_estimates():
    r = synthetic_cache()
    out = ana.noisy_mode(r, {}, "oracle_plane", np.random.default_rng(0))
    assert out["valid"].all()
    assert np.max(np.abs(out["rel_range_err"])) < 1e-9  # flat ground, exact plane


def test_injected_noise_matches_propagated_sigma_on_flat_ground():
    """End to end: z = error / predicted sigma has unit spread when the model is right."""
    r = synthetic_cache(n_frames=200)
    out = ana.noisy_mode(r, SIG, "oracle_plane", np.random.default_rng(1))
    z = out["range_err_m"][out["valid"]] / out["pred_range_sigma_m"][out["valid"]]
    assert np.std(z) == pytest.approx(1.0, abs=0.05)
    assert abs(np.mean(z)) < 0.05


def test_monte_carlo_sigma_agrees_with_first_order_in_linear_regime():
    r = synthetic_cache(n_frames=5, n_pix=40)
    idx = np.arange(len(r["u"]))
    fo = ana.first_order_sigma(r, idx, SIG, "oracle_plane")
    mc = ana.monte_carlo_sigma(r, idx, SIG, "oracle_plane", n_mc=3000, rng=np.random.default_rng(2))
    np.testing.assert_allclose(mc, fo, rtol=0.1)


def test_sensitivity_contributions_add_in_quadrature():
    r = synthetic_cache(n_frames=4, n_pix=50)
    contrib = ana.sensitivity(r, SIG, "oracle_plane")
    assert set(contrib) == set(SIG)
    idx = np.arange(len(r["u"]))
    total = ana.first_order_sigma(r, idx, SIG, "oracle_plane")
    per = np.stack([contrib[p]["per_sample"] for p in SIG], axis=1)
    np.testing.assert_allclose(np.sqrt((per**2).sum(axis=1)), total, rtol=1e-6)


def test_tilted_plane_mode_is_exact_on_a_tilted_plane_and_flat_is_not():
    n = (np.sin(15 * DEG), 0.0, -np.cos(15 * DEG))
    r = synthetic_cache(plane_n=n)
    tilted = ana.tilted_plane_mode(r)
    flat = ana.noisy_mode(r, {}, "oracle_plane", np.random.default_rng(0))
    assert np.max(np.abs(tilted["rel_range_err"][tilted["valid"]])) < 1e-6
    assert np.median(np.abs(flat["rel_range_err"][flat["valid"]])) > 0.01


def test_local_slope_plane_is_exact_when_terrain_is_one_plane():
    n = (np.sin(10 * DEG), np.sin(5 * DEG), -1.0)
    r = synthetic_cache(plane_n=n)
    loc = ana.local_slope_mode(r)
    assert np.max(np.abs(loc["rel_range_err"][loc["valid"]])) < 1e-6


def test_unknown_base_rejected():
    with pytest.raises(ValueError):
        ana.noisy_mode(synthetic_cache(n_frames=1), {}, "sky", np.random.default_rng(0))


def test_robust_plane_never_trims_below_three_points():
    # Three exact points plus one far outlier; trimming must keep a valid fit.
    X = np.array([[0, 0, 0.0], [1, 0, 0.0], [0, 1, 0.0], [5, 5, 50.0]])
    n, off = ana.robust_plane(X)
    assert np.all(np.isfinite(n))


def test_robust_plane_rejects_too_few_points():
    with pytest.raises(ValueError):
        ana.robust_plane(np.zeros((2, 3)))


def test_tilted_plane_mode_marks_frames_with_too_few_points_invalid():
    r = synthetic_cache(n_frames=2, n_pix=300)
    first = np.nonzero(r["frame"] == 0)[0]
    keep = np.r_[first[:2], np.nonzero(r["frame"] == 1)[0]]
    r2 = {k: v[keep] for k, v in r.items()}
    out = ana.tilted_plane_mode(r2)
    assert not out["valid"][:2].any()
    assert out["valid"][2:].all()
