"""Phase 14 radio and gossip (D-752)."""

from __future__ import annotations

import numpy as np
import pytest

from aerialrange.search import comms as cm
from aerialrange.search.belief import Detection, Observation


def _obs(drone, seq, cells=(1, 2, 3), pd=0.5, dets=()):
    return Observation(drone, seq, float(seq), np.array(cells), np.full(len(cells), pd), dets, 1e-4)


def test_observation_size_counts_runs_and_detections():
    assert cm.observation_bytes(_obs(0, 0, (5, 6, 7))) == cm.OBS_HEADER_BYTES + cm.RUN_BYTES
    two_runs = cm.observation_bytes(_obs(0, 0, (5, 6, 9)))
    assert two_runs == cm.OBS_HEADER_BYTES + 2 * cm.RUN_BYTES
    o = Observation(
        0, 0, 0.0, np.array([1, 2]), np.array([0.2, 0.9]), (Detection(0, 0, (1, 0, 0, 1)),), 1e-4
    )
    assert cm.observation_bytes(o) == cm.OBS_HEADER_BYTES + 2 * cm.RUN_BYTES + cm.DETECTION_BYTES


def test_vector_is_the_gap_free_prefix():
    n = cm.Node(0, 3)
    for s in (0, 1, 3):
        n.add(_obs(2, s))
    assert n.vector().tolist() == [0, 0, 2]
    assert n.add(_obs(2, 1)) is False


def _run(nodes, pos, radio, rounds, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        cm.exchange(nodes, pos, radio, 1.0, rng)


def test_everyone_gets_everything_without_loss():
    nodes = [cm.Node(i, 3) for i in range(3)]
    for i in range(3):
        for s in range(4):
            nodes[i].add(_obs(i, s))
    _run(nodes, np.zeros((3, 2)), cm.Radio(), 3)
    for nd in nodes:
        assert nd.vector().tolist() == [4, 4, 4]


def test_observations_are_relayed_beyond_radio_range():
    nodes = [cm.Node(i, 3) for i in range(3)]
    nodes[0].add(_obs(0, 0))
    pos = np.array([[0.0, 0.0], [0.0, 900.0], [0.0, 1800.0]])  # 0 and 2 out of range
    _run(nodes, pos, cm.Radio(range_m=1000.0), 4)
    assert nodes[2].vector().tolist() == [1, 0, 0]


def test_total_loss_delivers_nothing_and_partial_loss_delivers_eventually():
    for loss, ok in ((1.0, False), (0.7, True)):
        nodes = [cm.Node(i, 2) for i in range(2)]
        nodes[0].add(_obs(0, 0))
        _run(nodes, np.zeros((2, 2)), cm.Radio(loss=loss), 40, seed=3)
        assert bool(nodes[1].vector()[0] == 1) is ok


def test_budget_limits_what_is_sent_per_round():
    nodes = [cm.Node(i, 2) for i in range(2)]
    for s in range(10):
        nodes[0].add(_obs(0, s))
    per = cm.observation_bytes(_obs(0, 0))
    base = cm.HEADER_BYTES + cm.VECTOR_ENTRY_BYTES * 2
    radio = cm.Radio(budget_bytes=base + 3 * per)
    rng = np.random.default_rng(0)
    got = cm.exchange(nodes, np.zeros((2, 2)), radio, 1.0, rng)
    assert len(got[1]) == 3
    assert nodes[0].sent_bytes == base + 3 * per
    cm.exchange(nodes, np.zeros((2, 2)), radio, 1.0, rng)
    assert nodes[1].vector()[0] >= 3  # keeps going in later rounds
    silent = cm.Radio(budget_bytes=base - 1)
    before = nodes[0].sent_bytes
    cm.exchange(nodes, np.zeros((2, 2)), silent, 1.0, rng)
    assert nodes[0].sent_bytes == before


def test_loss_must_be_a_probability():
    with pytest.raises(ValueError):
        cm.Radio(loss=1.5)
