"""Local terrain surface from triangulated points and ray intersection (plan D), SYNTHETIC."""

import numpy as np
import pytest

from aerialrange.geometry.terrain import fit_local_surface, range_on_local_terrain

DEG = np.pi / 180
C = np.array([0.0, 0.0, -30.0])  # camera 30 m above the origin (NED)


def ray_to(P):
    d = P - C
    return d / np.linalg.norm(d)


def plane_points(n=300, slope_n=0.0, slope_e=0.0, z0=0.0, rough=0.0, seed=0, span=80):
    """Points on z = z0 + slope_n * north + slope_e * east (+ Gaussian roughness)."""
    rng = np.random.default_rng(seed)
    ne = rng.uniform(0, span, size=(n, 2)) + [0, -span / 2]
    z = (
        z0 + slope_n * ne[:, 0] + slope_e * ne[:, 1] + rng.normal(0, rough, n)
        if rough
        else z0 + slope_n * ne[:, 0] + slope_e * ne[:, 1]
    )
    return np.column_stack([ne, z])


def test_exact_on_a_tilted_plane():
    pts = plane_points(slope_n=-0.2, slope_e=0.1)  # rising to the north (Down decreases)
    target = np.array([40.0, 5.0, -0.2 * 40 + 0.1 * 5])
    r = range_on_local_terrain(C, ray_to(target), pts, radius_m=15.0, min_points=10)
    assert r["valid"], r["reason"]
    np.testing.assert_allclose(r["point_w"], target, atol=1e-6)
    assert r["range_m"] == pytest.approx(np.linalg.norm(target - C), rel=1e-9)
    assert r["slope_deg"] == pytest.approx(np.degrees(np.arctan(np.hypot(0.2, 0.1))), rel=1e-6)


def test_much_better_than_a_flat_plane_on_a_hill():
    rng = np.random.default_rng(1)
    ne = rng.uniform([0, -40], [80, 40], size=(600, 2))

    def hill(n, e):  # 6 m high Gaussian hill
        return -6.0 * np.exp(-((n - 45) ** 2 + e**2) / (2 * 15**2))

    pts = np.column_stack([ne, hill(ne[:, 0], ne[:, 1])])
    target = np.array([38.0, 4.0, hill(38.0, 4.0)])
    d = ray_to(target)
    r = range_on_local_terrain(C, d, pts, radius_m=8.0, min_points=10)
    flat_z = np.median(pts[:, 2])
    lam = (flat_z - C[2]) / d[2]
    flat_err = abs(lam - np.linalg.norm(target - C))
    assert abs(r["range_m"] - np.linalg.norm(target - C)) < 0.2 * flat_err


def test_outliers_above_ground_are_rejected():
    pts = plane_points(n=400, slope_n=-0.1)
    rng = np.random.default_rng(2)
    trees = pts[rng.choice(len(pts), 80, replace=False)].copy()
    trees[:, 2] -= rng.uniform(5, 15, len(trees))  # canopy points 5-15 m above ground
    allp = np.vstack([pts, trees])
    target = np.array([40.0, 0.0, -0.1 * 40])
    r = range_on_local_terrain(C, ray_to(target), allp, radius_m=15.0, min_points=10)
    assert np.linalg.norm(r["point_w"] - target) < 0.3


def test_too_few_points_is_invalid():
    r = range_on_local_terrain(
        C, ray_to(np.array([40.0, 0, 0])), plane_points(n=5), radius_m=5.0, min_points=10
    )
    assert not r["valid"] and r["reason"] == "too_few_points"


def test_reported_roughness_and_calibrated_range_sigma():
    """Bumpy ground with known roughness: the stated sigma should cover the truth ~68 %."""
    rough = 0.3
    zs, rs = [], []
    for seed in range(300):
        pts = plane_points(n=250, slope_n=-0.1, rough=rough, seed=seed)
        rng = np.random.default_rng(10_000 + seed)
        t_ne = np.array([40.0, rng.uniform(-5, 5)])
        target = np.array([*t_ne, -0.1 * t_ne[0] + rng.normal(0, rough)])
        r = range_on_local_terrain(C, ray_to(target), pts, radius_m=15.0, min_points=10)
        assert r["valid"]
        rs.append(r["roughness_m"])
        zs.append((r["range_m"] - np.linalg.norm(target - C)) / r["range_sigma_m"])
    # About 25 support points per patch, so single estimates scatter by roughly 15 %;
    # the estimator must be right on average.
    assert np.median(rs) == pytest.approx(rough, rel=0.1)
    zs = np.array(zs)
    assert np.mean(np.abs(zs) <= 1) == pytest.approx(0.68, abs=0.08)
    assert np.std(zs) == pytest.approx(1.0, abs=0.2)


def test_fit_local_surface_returns_height_and_variance():
    pts = plane_points(n=200, slope_n=0.05, z0=1.0)
    s = fit_local_surface(pts, center_ne=np.array([40.0, 0.0]), radius_m=20.0, min_points=10)
    assert s["valid"]
    assert s["height_at_center"] == pytest.approx(1.0 + 0.05 * 40, abs=1e-9)
    assert s["height_var_at_center"] >= 0
    assert s["n_support"] > 10


def test_k_nearest_mode_is_exact_on_a_plane_and_uses_k_points():
    pts = plane_points(n=300, slope_n=-0.15, slope_e=0.05)
    target = np.array([45.0, -3.0, -0.15 * 45 + 0.05 * -3])
    r = range_on_local_terrain(C, ray_to(target), pts, radius_m=None, min_points=10, k_nearest=25)
    assert r["valid"]
    np.testing.assert_allclose(r["point_w"], target, atol=1e-6)
    assert r["n_support"] == 25


def test_k_nearest_adapts_to_sparse_far_points():
    """Few points far away: a fixed small radius fails, k-nearest still fits."""
    rng = np.random.default_rng(4)
    near = plane_points(n=300, span=40, slope_n=-0.1)
    far_ne = rng.uniform([150, -30], [250, 30], size=(30, 2))
    far = np.column_stack([far_ne, -0.1 * far_ne[:, 0]])
    pts = np.vstack([near, far])
    target = np.array([200.0, 0.0, -20.0])
    fixed = range_on_local_terrain(C, ray_to(target), pts, radius_m=5.0, min_points=10)
    knn = range_on_local_terrain(C, ray_to(target), pts, radius_m=None, min_points=10, k_nearest=20)
    assert not fixed["valid"]
    assert knn["valid"]
    assert abs(knn["range_m"] - np.linalg.norm(target - C)) < 1e-6 * np.linalg.norm(target - C)


def test_initial_centre_selects_the_right_intersection_on_folded_terrain():
    """Two surfaces the ray could meet: a near steep bank and the true far ground. A start
    point near the far ground must converge there."""
    rng = np.random.default_rng(5)
    far_ne = rng.uniform([60, -10], [90, 10], size=(150, 2))
    far = np.column_stack([far_ne, np.zeros(150)])
    bank_ne = rng.uniform([5, -10], [15, 10], size=(150, 2))
    bank = np.column_stack([bank_ne, -3.0 * (bank_ne[:, 0] - 5)])  # rises steeply
    pts = np.vstack([far, bank])
    target = np.array([75.0, 0.0, 0.0])
    r = range_on_local_terrain(
        C,
        ray_to(target),
        pts,
        radius_m=None,
        min_points=10,
        k_nearest=30,
        init_center_ne=np.array([72.0, 1.0]),
    )
    assert r["valid"]
    np.testing.assert_allclose(r["point_w"], target, atol=1e-6)
