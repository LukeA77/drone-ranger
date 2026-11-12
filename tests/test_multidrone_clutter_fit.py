"""False-alarm recurrence measurement for Phase 16 (scripts/multidrone_clutter_fit.py, D-764)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multidrone_clutter_fit as cf  # noqa: E402


def test_two_kind_fit_recovers_known_values():
    rng = np.random.default_rng(0)
    sites = []
    for _ in range(4000):
        n, persistent = int(rng.integers(1, 5)), rng.random() < 0.3
        sites.append([bool(persistent and rng.random() < 0.6) for _ in range(n)])
    share, fire = cf.fit_persistent(sites)
    assert abs(share - 0.3) < 0.05 and abs(fire - 0.6) < 0.05


def test_two_kind_fit_with_no_recurrence_is_zero():
    assert cf.fit_persistent([[False, False], [False], []]) == (0.0, 0.0)
    assert cf.fit_persistent([]) == (0.0, 0.0)


def test_pass_index_finds_the_pass_holding_a_frame():
    passes = cf.split_passes(np.array([1, 2, 3, 50, 51, 120]), max_gap=10)
    assert len(passes) == 3
    assert cf.pass_index(passes, 2) == 0 and cf.pass_index(passes, 51) == 1
    assert cf.pass_index(passes, 60) == 1  # between passes: the last one started before it


def test_recurrence_counts_later_and_earlier_passes():
    # site of track 1 seen in frames 0-4, 40-44 and 80-84; track 2 near it on the second view
    sites = pd.DataFrame({"n": [0.0], "e": [0.0], "first": [1], "last": [3]}, index=[1])
    rows = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 3],
            "frame0": [1, 3, 41, 82],
            "ground_n_m": [0.0, 0.0, 0.5, 30.0],  # track 3 is far away
            "ground_e_m": [0.0, 0.0, 0.5, 30.0],
        }
    )
    seen = {1: np.r_[0:5, 40:45, 80:85]}
    later, repeat, per_site = cf.recurrence(sites, rows, seen, max_gap=5, radius_m=2.0)
    assert later == [True, False] and repeat == [False] and per_site == [[True, False]]
