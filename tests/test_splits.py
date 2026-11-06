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


# --- stratified group split (SynDrone towns) ---
from aerialrange.data.splits import split_stratified  # noqa: E402

STRATA = {
    "small": ["Town01", "Town02"],
    "city": ["Town03", "Town05", "Town10HD"],
    "open": ["Town04", "Town06", "Town07"],
}
QUOTA = {"development": 1, "test": 1}  # per stratum; the rest go to calibration


def test_stratified_split_one_per_stratum_and_rest():
    s = split_stratified(STRATA, QUOTA, rest="calibration", seed=0)
    assert set(s) == {"development", "test", "calibration"}
    for split in ("development", "test"):
        got = s[split]
        assert len(got) == 3
        for members in STRATA.values():
            assert len(set(got) & set(members)) == 1
    assert len(s["calibration"]) == 2
    everything = sorted(t for v in s.values() for t in v)
    assert everything == sorted(t for v in STRATA.values() for t in v)


def test_stratified_split_deterministic_and_order_free():
    a = split_stratified(STRATA, QUOTA, rest="calibration", seed=0)
    rev = {k: list(reversed(v)) for k, v in reversed(list(STRATA.items()))}
    assert a == split_stratified(rev, QUOTA, rest="calibration", seed=0)


def test_stratified_split_seed_matters():
    outs = {str(split_stratified(STRATA, QUOTA, rest="calibration", seed=k)) for k in range(10)}
    assert len(outs) > 1


def test_stratified_split_rejects_short_stratum_and_duplicates():
    with pytest.raises(ValueError):
        split_stratified({"a": ["x"]}, {"development": 1, "test": 1}, rest="cal", seed=0)
    with pytest.raises(ValueError):
        split_stratified({"a": ["x", "y"], "b": ["x", "z"]}, {"d": 1}, rest="c", seed=0)
