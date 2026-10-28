"""Statistics helpers on synthetic data with known answers."""

import numpy as np
import pytest
from scipy.stats import norm

from aerialrange.evaluate import stats as st

RNG = np.random.default_rng(0)


def test_group_bootstrap_ci_contains_truth_and_shrinks_with_more_groups():
    def make(n_groups):
        g = np.repeat(np.arange(n_groups), 200)
        offsets = RNG.normal(0, 1.0, n_groups)  # strong within-group correlation
        return offsets[g] + RNG.normal(0, 0.1, len(g)), g

    x10, g10 = make(10)
    x40, g40 = make(40)
    lo10, hi10 = st.group_bootstrap_ci(x10, g10, np.mean, n_boot=2000, rng=np.random.default_rng(1))
    lo40, hi40 = st.group_bootstrap_ci(x40, g40, np.mean, n_boot=2000, rng=np.random.default_rng(1))
    assert lo10 < 0 < hi10
    assert (hi40 - lo40) < (hi10 - lo10)


def test_group_bootstrap_is_wider_than_naive_sample_bootstrap_for_correlated_data():
    g = np.repeat(np.arange(8), 500)
    x = RNG.normal(0, 1.0, 8)[g] + RNG.normal(0, 0.1, len(g))
    lo, hi = st.group_bootstrap_ci(x, g, np.mean, n_boot=1000, rng=np.random.default_rng(2))
    naive = 2 * 1.96 * x.std() / np.sqrt(len(x))
    assert (hi - lo) > 5 * naive


def test_group_bootstrap_reproducible():
    x, g = RNG.normal(size=300), np.repeat(np.arange(6), 50)
    a = st.group_bootstrap_ci(x, g, np.median, 200, np.random.default_rng(3))
    b = st.group_bootstrap_ci(x, g, np.median, 200, np.random.default_rng(3))
    assert a == b


def test_binned_stat_masks_sparse_bins_and_counts_groups():
    x = np.array([0.5, 0.6, 1.5, 1.6, 1.7, 2.5])
    y = np.array([1.0, 3.0, 2.0, 4.0, 6.0, 9.0])
    g = np.array([0, 1, 0, 1, 2, 0])
    rows = st.binned_stat(
        x,
        y,
        g,
        edges=[0, 1, 2, 3],
        stat=np.median,
        n_boot=50,
        min_samples=2,
        rng=np.random.default_rng(0),
    )
    assert [r["n"] for r in rows] == [2, 3, 1]
    assert [r["n_groups"] for r in rows] == [2, 3, 1]
    assert rows[0]["stat"] == 2.0 and rows[1]["stat"] == 4.0
    assert rows[2]["reported"] is False and np.isnan(rows[2]["stat"])
    assert rows[1]["ci_lo"] <= rows[1]["stat"] <= rows[1]["ci_hi"]


def test_coverage_of_standard_normal_matches_nominal():
    z = np.random.default_rng(4).normal(size=200_000)
    cov = st.coverage(z, levels=(0.5, 0.68, 0.9, 0.95, 0.99))
    for p, c in cov.items():
        assert c == pytest.approx(p, abs=0.005)


def test_coverage_detects_overconfidence():
    z = 2.0 * np.random.default_rng(5).normal(size=50_000)  # sigma too small by 2x
    assert st.coverage(z, levels=(0.68,))[0.68] < 0.45


def test_sigma_scale_maximum_likelihood():
    rng = np.random.default_rng(6)
    sig = rng.uniform(0.5, 3.0, 100_000)
    e = 1.7 * sig * rng.normal(size=sig.size)
    assert st.fit_sigma_scale(e, sig) == pytest.approx(1.7, rel=0.01)


def test_interval_halfwidth_factor():
    assert st.z_halfwidth(0.95) == pytest.approx(norm.ppf(0.975))


def test_auroc_perfect_random_and_ties():
    y = np.array([0, 0, 1, 1])
    assert st.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert st.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert st.auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    s = np.random.default_rng(7).normal(size=20000)
    lab = np.random.default_rng(8).integers(0, 2, 20000)
    assert st.auroc(s, lab) == pytest.approx(0.5, abs=0.02)


def test_precision_recall_curve():
    s = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([1, 0, 1, 0])
    pr = st.precision_recall(s, y)
    # Thresholds in decreasing score order: flag top-1, top-2, top-3, all.
    np.testing.assert_allclose(pr["precision"], [1.0, 0.5, 2 / 3, 0.5])
    np.testing.assert_allclose(pr["recall"], [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(pr["threshold"], [0.9, 0.8, 0.7, 0.1])


def test_operating_point_picks_threshold_meeting_target_recall():
    s = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([1, 0, 1, 0])
    op = st.threshold_for_recall(s, y, target_recall=1.0)
    assert op["threshold"] == 0.7 and op["precision"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        st.threshold_for_recall(s, np.zeros(4, int), 0.5)


def test_stability_check_reports_relative_change():
    x, g = RNG.normal(size=600), np.repeat(np.arange(12), 50)
    r = st.bootstrap_stability(x, g, np.mean, n_boot=500, rng_seed=0)
    assert set(r) == {"n_boot", "ci", "ci_double", "max_endpoint_change"}
    assert r["max_endpoint_change"] >= 0


def test_binned_stat_requires_minimum_groups():
    x = np.array([0.5, 0.6, 0.7, 0.8])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array([0, 0, 1, 1])
    rows = st.binned_stat(
        x,
        y,
        g,
        edges=[0, 1],
        stat=np.median,
        n_boot=20,
        min_samples=2,
        rng=np.random.default_rng(0),
        min_groups=3,
    )
    assert rows[0]["reported"] is False


def test_usable_limit_from_increasing_x():
    # Error falls as depression rises: usable from the first bin whose CI upper bound, and
    # that of every bin above it, is below the threshold.
    rows = [
        {"lo": 0, "hi": 2, "reported": True, "ci_hi": 4.0},
        {"lo": 2, "hi": 5, "reported": True, "ci_hi": 0.3},
        {"lo": 5, "hi": 10, "reported": True, "ci_hi": 0.08},
        {"lo": 10, "hi": 20, "reported": True, "ci_hi": 0.05},
        {"lo": 20, "hi": 40, "reported": False, "ci_hi": float("nan")},
    ]
    assert st.usable_limit(rows, threshold=0.1, direction="above") == 5
    assert st.usable_limit(rows, threshold=0.01, direction="above") is None


def test_usable_limit_from_decreasing_x():
    # Error grows with range: usable below the first bin that crosses the threshold.
    rows = [
        {"lo": 1, "hi": 5, "reported": True, "ci_hi": 0.02},
        {"lo": 5, "hi": 10, "reported": True, "ci_hi": 0.05},
        {"lo": 10, "hi": 20, "reported": True, "ci_hi": 0.2},
        {"lo": 20, "hi": 50, "reported": True, "ci_hi": 0.08},
    ]
    assert st.usable_limit(rows, threshold=0.1, direction="below") == 10


def test_best_f1_threshold():
    s = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([1, 0, 1, 0])
    op = st.threshold_max_f1(s, y)
    assert op["threshold"] == 0.7 and op["f1"] == pytest.approx(0.8)
