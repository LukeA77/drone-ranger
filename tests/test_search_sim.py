"""Phase 14 simulation: footprint, lanes, sensor messages and whole searches (D-752)."""

from __future__ import annotations

import numpy as np
import pytest

from aerialrange.search import comms as cm
from aerialrange.search import sim
from aerialrange.search.grid import Grid
from aerialrange.search.prior import uniform_prior
from aerialrange.search.sensor import (
    SensorModel,
    observation_detections,
    tracked_object_array,
)


def _sensor(pd=1.0, fa=0.0, h=50):
    # test values only; the simulation's come from results/multi_drone/sensor_*.json
    return SensorModel(
        height_m=h,
        tilt_deg={20: 30.0, 50: 60.0, 80: 90.0}[h],
        pd_per_pass=pd,
        fa_per_m2=fa,
        pos_sigma_m=0.5,
        min_dep_deg=10.0,
    )


def test_footprint_lies_ahead_along_the_heading():
    s = _sensor()
    for yaw, axis, sign in ((0.0, 0, 1), (np.pi / 2, 1, 1), (np.pi, 0, -1)):
        fp = s.footprint(100.0, 200.0, yaw)
        centre = fp.mean(axis=0) - np.array([100.0, 200.0])
        assert np.sign(centre[axis]) == sign and abs(centre[axis]) > 10
    assert sim.lane_spacing_m(s) == pytest.approx(87, abs=1)  # SynDrone 50 m camera, near edge


def test_rotated_footprint_equals_the_ray_model():
    for h in (20, 50, 80):
        s = _sensor(h=h)
        for yaw in (0.0, 0.7, 2.0, -2.5):
            assert s.footprint(30.0, -40.0, yaw) == pytest.approx(
                s.exact_footprint(30.0, -40.0, yaw), abs=1e-6
            )


def test_lanes_cover_the_rectangle_without_gaps():
    pts = sim.lanes(0.0, 100.0, 0.0, 95.0, 20.0, (0.0, 0.0))
    es = sorted({p[1] for p in pts})
    assert len(es) == 5 and es[0] <= 10 and es[-1] >= 85
    assert max(np.diff(es)) <= 20.0
    assert pts[0] == (0.0, es[0])  # starts at the corner nearest the start


def test_detection_probability_follows_pass_length():
    s = SensorModel(
        height_m=20,
        tilt_deg=30.0,
        pd_per_pass=0.4,
        fa_per_m2=0.0,
        pos_sigma_m=0.4,
        min_dep_deg=10.0,
        pd_edges_frames=(27.0, 53.0, 84.0),
        pd_bins=(0.1, 0.4, 0.6),
    )
    assert s.pd_for(np.array([1, 25, 27, 28, 53, 84, 200])).tolist() == [
        0.1,
        0.1,
        0.1,
        0.4,
        0.4,
        0.6,
        0.6,
    ]
    assert _sensor().pd_for(np.array([5.0, 500.0])).tolist() == [1.0, 1.0]


def test_messages_round_trip_into_detections():
    msg = tracked_object_array(3.0, 4, [(10.0, 20.0)], 0.5, 7)
    assert msg["objects"][0]["position"] == {"x": 20.0, "y": 10.0, "z": 0.0}
    (d,) = observation_detections(msg)
    assert (d.n_m, d.e_m) == (10.0, 20.0) and d.cov[0] == pytest.approx(0.25)
    msg["objects"][0]["reliability"] = "uncertain"
    assert observation_detections(msg) == ()


def _scenario(n=2, pd=1.0, fa=0.0, horizon=4000.0):
    g = Grid.centred((0.0, 0.0), 300.0, 10.0)
    return sim.Scenario(g, uniform_prior(g), _sensor(pd, fa), n, 21.9, (0.0, 0.0), horizon)


@pytest.mark.parametrize("kind", sim.PLANNERS)
def test_a_certain_camera_finds_the_person(kind):
    sc = _scenario()
    for seed in range(3):
        r = sim.run(sc, sim.PlannerConfig(kind), cm.Radio(), seed)
        assert r.found and r.t_found_s > 0 and r.false_alarms_total == 0


def test_runs_repeat_exactly_for_a_seed_and_differ_between_seeds():
    sc = _scenario(pd=0.5, fa=2e-4)
    a = sim.run(sc, sim.PlannerConfig("auction"), cm.Radio(loss=0.3), 5)
    b = sim.run(sc, sim.PlannerConfig("auction"), cm.Radio(loss=0.3), 5)
    c = sim.run(sc, sim.PlannerConfig("auction"), cm.Radio(loss=0.3), 6)
    assert (a.t_found_s, a.false_alarms_total, a.bytes_sent) == (
        b.t_found_s,
        b.false_alarms_total,
        b.bytes_sent,
    )
    assert (a.t_found_s, a.cell_passes) != (c.t_found_s, c.cell_passes)


def test_sharing_sends_bytes_and_not_sharing_sends_none():
    sc = _scenario(pd=0.3)
    on = sim.run(sc, sim.PlannerConfig("greedy", share=True), cm.Radio(), 1)
    off = sim.run(sc, sim.PlannerConfig("greedy", share=False), cm.Radio(), 1)
    assert on.bytes_sent > 0 and off.bytes_sent == 0


def test_sharing_cuts_repeated_coverage():
    sc = _scenario(n=3, pd=0.3, horizon=400.0)
    rep = {True: [0, 0], False: [0, 0]}
    for share in (True, False):
        for seed in range(4):
            r = sim.run(sc, sim.PlannerConfig("auction", share=share), cm.Radio(), seed)
            rep[share][0] += r.repeat_cell_passes
            rep[share][1] += r.cell_passes
    assert rep[True][0] / rep[True][1] < rep[False][0] / rep[False][1]
