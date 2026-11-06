"""SynDrone evaluation helpers on small hand-made cases."""

import numpy as np
import pandas as pd
import pytest

from aerialrange.evaluate.syndrone_eval import (
    TruthRules,
    bootstrap,
    classify_truth,
    interval_coverage,
    keep_not_ignored,
    match_pairs,
)


def test_classify_truth():
    df = pd.DataFrame(
        {
            "tag": [40, 100, 105, 100, 40, 101],
            "level": [False, False, False, True, False, False],
            "visibility": [0.9, 0.2, 1.0, 1.0, 0.0, 0.8],
            "n_instance_px": [50, 80, 40, 60, 0, 10],
        }
    )
    out = classify_truth(df, TruthRules(min_visibility=0.5, min_instance_px=20))
    assert out["group"].tolist() == ["person", "vehicle", None, "vehicle", "person", "vehicle"]
    assert out["status"].tolist() == ["valid", "ignore", "ignore", "ignore", "absent", "ignore"]


def test_predictions_on_ignored_boxes_are_removed_but_not_on_valid_ones():
    valid = np.array([[0, 0, 10, 10.0]])
    ignored = np.array([[11, 0, 10, 10.0], [100, 100, 5, 5.0]])
    pred = np.array([[0, 0, 10, 10.0], [11, 0, 10, 10.0], [50, 50, 10, 10.0], [101, 100, 5, 5.0]])
    assert keep_not_ignored(pred, valid, ignored).tolist() == [True, False, True, False]
    # A prediction overlapping both a valid and an ignored box goes to the better match.
    p2 = np.array([[1, 0, 10, 10.0]])
    assert keep_not_ignored(p2, valid, ignored).tolist() == [True]


def test_match_pairs_one_to_one():
    gt = np.array([[0, 0, 10, 10.0], [20, 0, 10, 10.0]])
    pr = np.array([[21, 0, 10, 10.0], [0, 1, 10, 10.0], [200, 0, 5, 5.0]])
    m = sorted(match_pairs(gt, pr))
    assert [(a, b) for a, b, _ in m] == [(0, 1), (1, 0)]


def test_interval_coverage_gaussian():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 2.0, 200000)
    cov = interval_coverage(e, np.full_like(e, 2.0))
    assert cov["0.68"] == pytest.approx(0.68, abs=0.005)
    assert cov["0.95"] == pytest.approx(0.95, abs=0.005)


def test_bootstrap_resamples_units():
    units = [np.ones(10), np.ones(10) * 3]
    lo, hi = bootstrap(units, lambda us: float(np.mean(np.concatenate(us))), 2000, 0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(3.0)


def test_keep_not_ignored_prefers_valid_truth_on_exact_ties():
    """A map box exactly on a valid actor must not swallow the prediction (D-711)."""
    import numpy as np

    from aerialrange.evaluate.syndrone_eval import keep_not_ignored

    box = np.array([[100.0, 100.0, 40.0, 20.0]])
    for order in range(2):  # whichever way the solver would break the tie
        ignored = np.vstack([box, [[500.0, 500.0, 10.0, 10.0]]])[:: 1 - 2 * order]
        assert keep_not_ignored(box, box, ignored).tolist() == [True]
    # a prediction on an ignored box with no valid twin is still dropped
    assert keep_not_ignored(box, np.zeros((0, 4)), box).tolist() == [False]


def test_walkers_stacked_on_one_spot_are_ignored():
    """Several walkers at one ground position (a spawn glitch, D-712) are not people."""
    import pandas as pd

    from aerialrange.evaluate.syndrone_eval import TruthRules, classify_truth

    base = {"tag": 40, "level": False, "n_instance_px": 80, "visibility": 1.0, "height": 20}
    rows = [
        {**base, "frame": 0, "id": 1, "foot_x": 5.0, "foot_y": 5.0},
        {**base, "frame": 0, "id": 2, "foot_x": 5.0, "foot_y": 5.0},  # stacked on id 1
        {**base, "frame": 0, "id": 3, "foot_x": 5.3, "foot_y": 5.0},  # 0.3 m away: a person
        {**base, "frame": 1, "id": 1, "foot_x": 9.0, "foot_y": 5.0},  # walked off: a person
        {**base, "frame": 1, "id": 2, "foot_x": 5.0, "foot_y": 5.0},
        {**base, "frame": 0, "id": 9, "tag": 100, "foot_x": 5.0, "foot_y": 5.0},  # a car
    ]
    out = classify_truth(pd.DataFrame(rows), TruthRules(0.5, 4))
    assert out["status"].tolist() == ["ignore", "ignore", "valid", "valid", "valid", "valid"]
