"""Phase 14 core: grid, ring prior and the search map update (D-752)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aerialrange.search import prior as pr
from aerialrange.search.belief import Detection, Observation, SearchMap, log_likelihood
from aerialrange.search.grid import Grid


def test_grid_cells_and_centres_agree():
    g = Grid(100.0, -50.0, 10.0, 4, 5)
    n, e = g.centres()
    assert (g.cell_of(n, e) == np.arange(g.size)).all()
    assert g.cell_of(99.9, 0.0) == -1 and g.cell_of(100.0, -50.0) == 0
    assert g.cell_of(139.99, -0.01) == 3 * 5 + 4


def test_polygon_cells_use_the_centre_rule():
    g = Grid(0.0, 0.0, 1.0, 10, 10)
    # square from (2, 3) to (5, 7) north/east covers centres n 2.5..4.5, e 3.5..6.5
    poly = np.array([[2.0, 3.0], [2.0, 7.0], [5.0, 7.0], [5.0, 3.0]])
    got = set(g.cells_in_polygon(poly).tolist())
    want = {r * 10 + c for r in range(2, 5) for c in range(3, 7)}
    assert got == want
    tri = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]])  # n + e < 10
    n, e = g.centres()
    inside = set(np.nonzero(n + e < 10)[0].tolist())
    got = set(g.cells_in_polygon(tri).tolist())
    assert got == inside
    assert g.cells_in_polygon(np.array([[50.0, 50.0], [50.0, 60.0], [60.0, 60.0]])).size == 0


def test_ring_prior_holds_the_ring_shares():
    radii = (100.0, 300.0, 600.0, 1500.0)
    g = Grid.centred((0.0, 0.0), 2000.0, 5.0)
    p = pr.ring_prior(g, (0.0, 0.0), radii)
    n, e = g.centres()
    d = np.hypot(n, e)
    for r, share in zip(radii, np.cumsum(pr.RING_SHARES), strict=True):
        assert p[d < r].sum() == pytest.approx(share, abs=0.01)
    out = pr.OUTSIDE_SHARE / (pr.OUTSIDE_REGION_M**2 - np.pi * radii[-1] ** 2)
    assert p[d > radii[-1]].sum() == pytest.approx(out * (16e6 - np.pi * radii[-1] ** 2), rel=0.02)
    with pytest.raises(ValueError):
        pr.ring_density_per_m2(np.array([1.0]), (3.0, 2.0, 4.0, 5.0))


def test_koester_hiker_values_are_kilometres():
    km = np.array(pr.HIKER_TEMPERATE_MOUNTAIN_KM)
    miles_in_sava = np.array([0.7, 1.9, 3.6, 11.3])
    assert np.allclose(miles_in_sava * 1.609344, km, atol=0.15)


def test_sampled_locations_follow_the_prior():
    g = Grid(0.0, 0.0, 10.0, 2, 2)
    p = np.array([0.7, 0.1, 0.1, 0.1])
    rng = np.random.default_rng(0)
    pts = np.array([pr.sample_location(p, g, rng) for _ in range(4000)])
    share = np.bincount(g.cell_of(pts[:, 0], pts[:, 1]), minlength=4) / len(pts)
    assert share == pytest.approx(p, abs=0.03)


def _map(p):
    g = Grid(0.0, 0.0, 10.0, 1, len(p))
    return g, SearchMap.from_prior(g, np.array(p, float))


def test_a_look_without_detection_is_koopman_update():
    g, m = _map([0.2, 0.3, 0.1])  # 0.4 outside
    m.apply(Observation(0, 0, 0.0, np.array([0, 1]), np.array([0.5, 0.8])))
    w = np.array([0.2 * 0.5, 0.3 * 0.2, 0.1])
    z = w.sum() + 0.4
    p, out = m.probabilities()
    assert p == pytest.approx(w / z)
    assert out == pytest.approx(0.4 / z)


def test_a_certain_look_clears_cells():
    g, m = _map([0.5, 0.5])
    m.apply(Observation(0, 0, 0.0, np.array([0]), np.array([1.0])))
    p, _ = m.probabilities()
    assert p[0] == 0 and p[1] == pytest.approx(1.0)


def test_detection_likelihood_matches_the_formula():
    g = Grid(0.0, 0.0, 10.0, 1, 2)
    det = Detection(5.0, 5.0, (1.0, 0.0, 0.0, 1.0))  # centre of cell 0, sigma 1 m
    lam = 1e-4  # false alarms per m^2
    obs = Observation(0, 0, 0.0, np.array([0, 1]), np.array([0.6, 0.6]), (det,), lam)
    ll = np.exp(log_likelihood(g, obs))
    from scipy.stats import norm

    mass0 = (norm.cdf(5) - norm.cdf(-5)) ** 2  # all of the error inside cell 0
    mass1 = (norm.cdf(15 - 5) - norm.cdf(10 - 5)) * (norm.cdf(5) - norm.cdf(-5))
    assert ll[0] == pytest.approx(0.4 + 0.6 * mass0 / 100.0 / lam)
    assert ll[1] == pytest.approx(0.4 + 0.6 * mass1 / 100.0 / lam)
    with pytest.raises(ValueError):
        log_likelihood(g, Observation(0, 0, 0.0, np.array([0]), np.array([0.5]), (det,), -1.0))


def test_without_false_alarms_a_detection_rules_out_everything_else():
    g = Grid(0.0, 0.0, 10.0, 1, 3)
    m = SearchMap.from_prior(g, np.array([0.2, 0.2, 0.2]))
    det = Detection(5.0, 5.0, (1.0, 0.0, 0.0, 1.0))
    m.apply(Observation(0, 0, 0.0, np.array([0, 1]), np.array([0.6, 0.6]), (det,), 0.0))
    p, out = m.probabilities()
    assert out == 0 and p[2] == 0 and p[0] == pytest.approx(1.0, abs=1e-6)


def test_a_detection_raises_its_cell_less_when_false_alarms_are_common():
    probs = []
    for lam in (1e-5, 1e-3):
        g, m = _map([0.25, 0.25, 0.25, 0.25])
        det = Detection(5.0, 15.0, (0.25, 0.0, 0.0, 0.25))
        m.apply(Observation(0, 0, 0.0, np.arange(4), np.full(4, 0.5), (det,), lam))
        probs.append(m.probabilities()[0][1])
    assert probs[0] > probs[1] > 0.25


def test_merging_is_order_free_and_idempotent():
    g = Grid(0.0, 0.0, 10.0, 3, 3)
    rng = np.random.default_rng(1)
    obs = []
    for k in range(5):
        cells = rng.choice(9, size=4, replace=False)
        dets = (Detection(float(rng.uniform(0, 30)), float(rng.uniform(0, 30)), (4, 0, 0, 4)),)
        obs.append(Observation(k % 2, k, float(k), cells, rng.uniform(0.1, 0.9, 4), dets, 1e-3))
    ref = None
    for order in itertools.islice(itertools.permutations(obs), 0, 120, 7):
        m = SearchMap.from_prior(g, np.full(9, 1 / 9))
        for o in order + order[:2]:  # two arrive twice
            m.apply(o)
        p = m.probabilities()[0]
        if ref is None:
            ref = p
        assert p == pytest.approx(ref, rel=1e-12)
    assert m.apply(obs[0]) is False


def test_observation_must_not_repeat_cells():
    g, m = _map([0.5, 0.5])
    with pytest.raises(ValueError):
        m.apply(Observation(0, 0, 0.0, np.array([0, 0]), np.array([0.5, 0.5])))


def test_nearby_cells_only_gives_the_full_likelihood():
    from aerialrange.search import belief as bl

    g = Grid(0.0, 0.0, 10.0, 20, 20)
    rng = np.random.default_rng(4)
    cells = rng.choice(g.size, 150, replace=False)
    dets = tuple(
        Detection(float(rng.uniform(0, 200)), float(rng.uniform(0, 200)), (s * s, 0.0, 0.0, s * s))
        for s in (0.4, 3.0, 12.0)
    )
    obs = Observation(0, 0, 0.0, cells, rng.uniform(0.05, 0.9, len(cells)), dets, 2e-4)
    full = np.zeros(len(cells))
    for d in dets:
        full += bl.detection_density(g, cells, d)
    want = np.log((1 - obs.pd) + obs.pd * full / obs.fa_per_m2)
    assert bl._log_likelihood(g, obs) == pytest.approx(want, rel=1e-6, abs=1e-9)
