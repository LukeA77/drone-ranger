import pytest

from aerialrange.data.splits import allocate_counts, split_trajectories

NAMES = [f"trajectory_{i:04d}" for i in range(30)]
PROPS = {"development": 0.5, "calibration": 0.2, "test": 0.3}


def test_counts_follow_proportions():
    assert allocate_counts(30, PROPS) == {"development": 15, "calibration": 6, "test": 9}


def test_counts_largest_remainder_sums_to_total():
    c = allocate_counts(7, PROPS)
    assert sum(c.values()) == 7


def test_proportions_must_sum_to_one():
    with pytest.raises(ValueError):
        allocate_counts(10, {"a": 0.5, "b": 0.4})


def test_split_is_disjoint_complete_and_deterministic():
    a = split_trajectories(NAMES, PROPS, seed=0)
    b = split_trajectories(list(reversed(NAMES)), PROPS, seed=0)
    assert a == b  # independent of input order
    all_names = [n for v in a.values() for n in v]
    assert sorted(all_names) == NAMES
    assert len(set(all_names)) == len(all_names)
    assert {k: len(v) for k, v in a.items()} == {"development": 15, "calibration": 6, "test": 9}


def test_different_seed_changes_split():
    assert split_trajectories(NAMES, PROPS, seed=0) != split_trajectories(NAMES, PROPS, seed=1)
