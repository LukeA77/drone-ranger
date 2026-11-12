"""Phase 16 simulation (D-764): equal to Phase 14 in its settings; confirmation, recurring false
alarms, compact sharing and the sweep planner."""

from __future__ import annotations

import numpy as np
import pytest

from aerialrange.search import comms as cm
from aerialrange.search import compact as cp
from aerialrange.search import engine as en
from aerialrange.search import sim
from aerialrange.search import sweep as sw
from aerialrange.search.confirm import Alarms, ClutterSites
from aerialrange.search.grid import Grid
from aerialrange.search.prior import uniform_prior
from aerialrange.search.sensor import SensorModel


def _sensor(pd=1.0, fa=0.0, h=20):
    # test values only; the simulation's come from results/multi_drone/sensor_*.json
    return SensorModel(
        height_m=h,
        tilt_deg={20: 30.0, 50: 60.0, 80: 90.0}[h],
        pd_per_pass=pd,
        fa_per_m2=fa,
        pos_sigma_m=0.4,
        min_dep_deg=10.0,
    )


def _scenario(n=2, pd=1.0, fa=0.0, horizon=3000.0, h=20, half=300.0):
    g = Grid.centred((0.0, 0.0), half, 10.0)
    return sim.Scenario(g, uniform_prior(g), _sensor(pd, fa, h), n, 21.9, (0.0, 0.0), horizon)


# --- the new engine gives Phase 14's results in Phase 14's settings ---------------------------


def _same(a: sim.Result, b: en.Result) -> None:
    assert (a.found, a.t_found_s) == (b.found, b.t_found_s)
    assert (a.false_alarms_before_find, a.false_alarms_total) == (
        b.false_alarms_before_find,
        b.false_alarms_total,
    )
    assert (a.cell_passes, a.repeat_cell_passes, a.bytes_sent) == (
        b.cell_passes,
        b.repeat_cell_passes,
        b.bytes_sent,
    )


@pytest.mark.parametrize("kind", sim.PLANNERS)
@pytest.mark.parametrize("share", [True, False])
def test_engine_reproduces_phase14_exactly(kind, share):
    sc = _scenario(n=3, pd=0.5, fa=2e-4, horizon=900.0)
    radio = cm.Radio(loss=0.3, range_m=400.0)
    for seed in range(3):
        a = sim.run(sc, sim.PlannerConfig(kind, share), radio, seed)
        b = en.run(sc, en.Plan(kind, share), radio, seed)
        _same(a, b)


def test_engine_reproduces_phase14_under_a_byte_budget():
    sc = _scenario(n=3, pd=0.5, fa=2e-4, horizon=600.0)
    radio = cm.Radio(budget_bytes=400.0)
    for seed in range(2):
        _same(
            sim.run(sc, sim.PlannerConfig("auction"), radio, seed),
            en.run(sc, en.Plan("auction"), radio, seed),
        )


# --- confirmation and recurring false alarms --------------------------------------------------


def test_an_alarm_needs_k_detections_within_the_radius():
    a = Alarms(k=2, radius_m=2.0, cell_m=10.0)
    person = (100.0, 100.0)
    assert a.add(0.0, 0.0, 1.0, person) is None
    assert a.add(5.0, 0.0, 2.0, person) is None  # too far from the first
    assert a.add(1.0, 1.0, 3.0, person) is False  # second within 2 m of (0, 0): false alarm
    assert a.add(0.5, 0.5, 4.0, person) is None  # same place: no second alarm
    assert a.add(100.0, 100.0, 5.0, person) is None
    assert a.add(101.0, 100.5, 6.0, person) is True  # the person, confirmed
    assert [r[3] for r in a.raised] == [False, True]


def test_detections_across_a_cell_border_count_together():
    a = Alarms(k=2, radius_m=2.0, cell_m=10.0)
    assert a.add(9.5, 9.5, 1.0, (0.0, 0.0)) is None
    assert a.add(10.5, 10.5, 2.0, (50.0, 50.0)) is False


def test_k_one_raises_an_alarm_at_every_new_place():
    a = Alarms(k=1, radius_m=2.0, cell_m=10.0)
    assert a.add(0.0, 0.0, 1.0, (0.0, 0.0)) is True


def test_lookalikes_fire_with_their_probability():
    g = Grid.centred((0.0, 0.0), 100.0, 10.0)
    rng = np.random.default_rng(0)
    never, always = ClutterSites(g, 0.0), ClutterSites(g, 1.0)
    for s in (never, always):
        s.add(5.0, 5.0)
    c = np.array([int(g.cell_of(5.0, 5.0))])
    assert never.fire(c, 0.4, rng) == [] and len(always.fire(c, 0.4, rng)) == 1
    half = ClutterSites(g, 0.5)
    half.add(5.0, 5.0)
    hits = sum(len(half.fire(c, 0.4, rng)) for _ in range(4000))
    assert 1800 < hits < 2200
    assert always.fire(np.array([0]), 0.4, rng) == []  # another cell: nothing


def test_lookalikes_are_placed_at_their_density():
    g = Grid.centred((0.0, 0.0), 500.0, 10.0)  # 1 km2
    s = ClutterSites(g, 0.5)
    k = s.place(2e-4, np.random.default_rng(3))  # 200 expected
    assert 150 < k < 250 and sum(len(v) for v in s.sites.values()) == k
    assert ClutterSites(g, 0.5).place(0.0, np.random.default_rng(3)) == 0


def test_lookalikes_keep_the_total_false_alarm_rate():
    # one sweep over new ground: the share from look-alikes replaces one-off alarms, so the
    # total stays about the same; flying the same ground again does not add look-alikes
    sc = _scenario(n=1, pd=0.0, fa=3e-4, horizon=900.0)
    w0 = en.World(stop="confirmed")
    w = en.World(persistent_share=0.3, persistent_fire=0.6, stop="confirmed")
    a = [en.run(sc, en.Plan("lawnmower"), cm.Radio(), s, w0).false_alarms_total for s in range(6)]
    b = [en.run(sc, en.Plan("lawnmower"), cm.Radio(), s, w).false_alarms_total for s in range(6)]
    assert 0.8 < np.mean(b) / np.mean(a) < 1.2


def test_persistent_lookalikes_reach_the_crew():
    sc = _scenario(n=1, pd=0.0, fa=3e-4, horizon=600.0)
    w = en.World(persistent_share=0.5, persistent_fire=1.0, stop="confirmed")
    r = en.run(sc, en.Plan("lawnmower"), cm.Radio(), 2, w)
    r0 = en.run(sc, en.Plan("lawnmower"), cm.Radio(), 2, en.World(stop="confirmed"))
    assert r.crew_alarms_total > r0.crew_alarms_total


def test_confirmed_find_comes_after_the_first_detection():
    sc = _scenario(n=2, pd=1.0)
    for seed in range(3):
        r = en.run(sc, en.Plan("lawnmower"), cm.Radio(), seed, en.World(stop="confirmed"))
        assert r.confirmed and r.t_confirmed_s >= r.t_found_s
        assert r.crew_alarms_before_confirmed == 0


# --- compact sharing --------------------------------------------------------------------------


def test_quantised_pose_round_trips_within_half_a_quantum():
    for n, e, y in ((0.0, 0.0, 0.0), (1234.56, -987.65, 5.5), (-3000.0, 3000.0, -1.0)):
        qn, qe, qy = cp.quantise(n, e, y)
        a, b, c = cp.dequantise(qn, qe, qy)
        assert abs(a - n) <= 0.05 + 1e-9 and abs(b - e) <= 0.05 + 1e-9
        assert abs((c - y + np.pi) % (2 * np.pi) - np.pi) <= cp.YAW_Q
    with pytest.raises(ValueError):
        cp.quantise(4000.0, 0.0, 0.0)


def _logs(n_drones=3, steps=40):
    """Entries for drones flying straight lines, with one detection each now and then."""
    out = []
    for i in range(n_drones):
        log = []
        for k in range(steps):
            pos = (-200.0 + 21.9 * k, -150.0 + 120.0 * i)
            dets = [(pos[0] + 40.0, pos[1] + 3.0)] if k % 7 == 3 else []
            log.append(cp.make_entry(i, k, k, pos, 0.0, dets))
        out.append(log)
    return out


def test_replay_gives_the_same_observations_on_every_drone():
    sc = _scenario()
    logs = _logs()
    a = [cp.Replayer(sc.grid, sc.sensor, 1.0) for _ in logs]
    b = [cp.Replayer(sc.grid, sc.sensor, 1.0) for _ in logs]
    for i, log in enumerate(logs):
        for e in log:
            x, y = a[i].apply(e), b[i].apply(e)
            if x is None:
                assert y is None
                continue
            assert x.id == y.id and np.array_equal(x.cells, y.cells)
            assert np.array_equal(x.pd, y.pd) and x.detections == y.detections
    with pytest.raises(ValueError):
        cp.Replayer(sc.grid, sc.sensor, 1.0).apply(logs[0][1])  # out of order


def test_detections_survive_encoding_to_a_tenth_of_a_metre():
    e = cp.make_entry(0, 0, 0, (10.04, -5.02), 0.3, [(52.33, 7.71)])
    (d,) = e.detections_ne()
    assert abs(d[0] - 52.33) <= 0.1 and abs(d[1] - 7.71) <= 0.1


def _gossip(logs, radio, rounds_after=60, seed=0, pos=None):
    n = len(logs)
    nodes = [cp.CompactNode(i, n) for i in range(n)]
    rng = np.random.default_rng(seed)
    steps = len(logs[0])
    pos = np.zeros((n, 2)) if pos is None else pos
    sends = 0
    for k in range(steps + rounds_after):
        if k < steps:
            for i in range(n):
                nodes[i].add(logs[i][k])
        before = [x.sent_bytes for x in nodes]
        cp.exchange(nodes, [True] * n, pos, radio, k, 1.0, rng)
        sends += sum(x.sent_bytes - b for x, b in zip(nodes, before, strict=True))
    return nodes, sends


@pytest.mark.parametrize(
    "radio", [cm.Radio(), cm.Radio(loss=0.5), cm.Radio(loss=0.9), cm.Radio(budget_bytes=30.0)]
)
def test_every_drone_ends_with_every_entry(radio):
    logs = _logs()
    nodes, _ = _gossip(logs, radio, rounds_after=300)
    for x in nodes:
        assert x.prefix.tolist() == [len(log) for log in logs]


def test_entries_reach_drones_out_of_each_others_range_by_relay():
    logs = _logs()
    pos = np.array([[0.0, 0.0], [0.0, 400.0], [0.0, 800.0]])  # 0 and 2 never hear each other
    nodes, _ = _gossip(logs, cm.Radio(range_m=500.0), pos=pos)
    assert nodes[2].prefix[0] == len(logs[0]) and nodes[0].prefix[2] == len(logs[2])


def test_no_echo_each_entry_is_sent_once_when_all_hear_each_other():
    logs = _logs()
    nodes = [cp.CompactNode(i, 3) for i in range(3)]
    rng = np.random.default_rng(1)
    sent = 0
    orig = cp.CompactNode.wanted

    def counting(self, nb, rnd):
        nonlocal sent
        out = orig(self, nb, rnd)
        sent += len(out)
        return out

    cp.CompactNode.wanted = counting
    try:
        for k in range(len(logs[0]) + 10):
            if k < len(logs[0]):
                for i in range(3):
                    nodes[i].add(logs[i][k])
            cp.exchange(nodes, [True] * 3, np.zeros((3, 2)), cm.Radio(), k, 1.0, rng)
    finally:
        cp.CompactNode.wanted = orig
    assert sent == sum(len(log) for log in logs)


def test_a_drone_out_of_the_team_sends_nothing_and_hears_nothing():
    logs = _logs()
    nodes = [cp.CompactNode(i, 3) for i in range(3)]
    rng = np.random.default_rng(0)
    for k in range(10):
        for i in range(3):
            nodes[i].add(logs[i][k])
        cp.exchange(nodes, [True, True, False], np.zeros((3, 2)), cm.Radio(), k, 1.0, rng)
    assert nodes[2].sent_bytes == 0 and nodes[2].prefix.tolist() == [0, 0, 10]
    assert nodes[0].prefix[2] == 0


# --- the sweep planner ------------------------------------------------------------------------


def test_lanes_split_the_area_into_one_strip_per_drone():
    g = Grid.centred((0.0, 0.0), 300.0, 10.0)
    lay = sw.layout(g, 3, 40.0, (0.0, 0.0), 8)
    for s in range(3):
        es = [lay.lanes[i].e_m for i in lay.strip_lanes(s)]
        lo, hi = -300.0 + 200.0 * s, -300.0 + 200.0 * (s + 1)
        assert all(lo < e < hi for e in es) and max(np.diff(sorted(es))) <= 40.0
    # each strip starts from the side nearer the launch point (east 0)
    assert abs(lay.lanes[lay.strip_lanes(0)[0]].e_m) < abs(lay.lanes[lay.strip_lanes(0)[-1]].e_m)


def test_lane_choice_own_strip_first_then_the_far_end_of_the_most_behind():
    g = Grid.centred((0.0, 0.0), 300.0, 10.0)
    lay = sw.layout(g, 2, 40.0, (0.0, 0.0), 8)
    s0, s1 = lay.strip_lanes(0), lay.strip_lanes(1)
    assert sw.choose_lane(lay, 0, set(range(len(lay.lanes)))) == s0[0]
    assert sw.choose_lane(lay, 0, set(s1)) == s1[-1]
    assert sw.choose_lane(lay, 0, set()) is None


def test_a_silent_drone_is_presumed_lost_only_after_the_timeout():
    node = cp.CompactNode(0, 3)
    node.add(cp.make_entry(1, 0, 9, (0.0, 0.0), 0.0, [], task=4))
    live = sw.live(node, 20.0, 1.0, 30.0, 0)
    assert live == {1: 4, 2: cp.NO_TASK}  # drone 2 never heard: not presumed lost
    assert sw.live(node, 50.0, 1.0, 30.0, 0) == {2: cp.NO_TASK}


def test_finished_tasks_are_known_from_the_log():
    node = cp.CompactNode(0, 2)
    node.add(cp.make_entry(1, 0, 0, (0.0, 0.0), 0.0, [], task=3 | cp.DONE_BIT))
    assert 3 in node.done and sw.live(node, 1.0, 1.0, 30.0, 0) == {1: cp.NO_TASK}


@pytest.mark.parametrize("coordinator", [False, True])
def test_a_certain_camera_finds_the_person_with_the_sweep(coordinator):
    sc = _scenario(n=3, pd=1.0)
    plan = en.Plan("sweep", sharing="compact", coordinator=coordinator)
    for seed in range(3):
        r = en.run(sc, plan, cm.Radio(), seed, en.World(stop="confirmed"))
        assert r.found and r.confirmed and r.bytes_sent > 0


def test_a_lost_drones_strip_is_still_searched():
    # the person is placed by the seed; with a certain camera and one drone lost at once,
    # the others take its lanes after the timeout and find the person wherever they are
    sc = _scenario(n=3, pd=1.0, horizon=6000.0)
    plan = en.Plan("sweep", sharing="compact", lost_after_s=10.0)
    for seed in range(4):
        r = en.run(sc, plan, cm.Radio(), seed, en.World(lose=(1, 5.0), stop="first"))
        assert r.found


def test_without_the_lost_drones_lanes_the_lawnmower_can_miss():
    sc = _scenario(n=3, pd=1.0, horizon=2000.0)
    missed = 0
    for seed in range(12):
        r = en.run(sc, en.Plan("lawnmower"), cm.Radio(), seed, en.World(lose=(1, 5.0)))
        missed += not r.found
    assert missed > 0


def test_compact_sharing_sends_far_fewer_bytes_than_phase14():
    sc = _scenario(n=3, pd=0.4, fa=3e-4, horizon=600.0)
    a = en.run(sc, en.Plan("auction"), cm.Radio(), 3, en.World(stop="confirmed"))
    b = en.run(sc, en.Plan("auction", sharing="compact"), cm.Radio(), 3, en.World(stop="confirmed"))
    assert b.bytes_sent < a.bytes_sent / 3


def test_plan_settings_are_checked():
    with pytest.raises(ValueError):
        en.Plan("sweep")  # needs compact logs
    with pytest.raises(ValueError):
        en.Plan("auction", coordinator=True)
    with pytest.raises(ValueError):
        en.World(stop="never")
    with pytest.raises(ValueError):
        en.World(persistent_share=0.2)  # look-alikes that never fire


def test_rounds_follow_one_another_and_stop_when_not_resweeping():
    g = Grid.centred((0.0, 0.0), 300.0, 10.0)
    lay = sw.layout(g, 2, 40.0, (0.0, 0.0), 8)
    node = cp.CompactNode(0, 2)
    assert sw.current_round(lay, node, {}, 8)[0] == 0
    node.done |= {lay.lane_task(i, 0) for i in range(len(lay.lanes))}
    r, free = sw.current_round(lay, node, {}, 8)
    assert r == 1 and free == set(range(len(lay.lanes)))
    assert sw.current_round(lay, node, {}, 1) is None  # "map" and "coverage": one sweep only
    t = lay.lane_task(3, 2)
    assert lay.is_lane(t) and lay.lane_of(t) == 3 and lay.round_of(t) == 2
    b = lay.block_task(5)
    assert not lay.is_lane(b) and lay.block_of(b) == 5


@pytest.mark.parametrize("post", ["resweep", "map", "coverage"])
def test_every_post_sweep_option_finds_the_person(post):
    sc = _scenario(n=2, pd=0.6, fa=2e-4, horizon=5000.0)
    plan = en.Plan("sweep", sharing="compact", post=post)
    w = en.World(persistent_share=0.5, persistent_fire=0.2, stop="confirmed")
    found = sum(en.run(sc, plan, cm.Radio(), s, w).confirmed for s in range(4))
    assert found >= 3
