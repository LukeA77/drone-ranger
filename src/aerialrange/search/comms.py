"""Radio and observation sharing between drones (Phase 14, D-752).

Radio: a packet reaches every other drone within `range_m` (straight-line distance), each
copy lost independently with probability `loss`. Each drone may send at most `budget_bytes`
per second; a budget of None means no limit.

Sharing is anti-entropy gossip (Demers et al., "Epidemic algorithms for replicated database
maintenance", PODC 1987): every broadcast carries the sender's version vector (for each drone,
how many of its observations, numbered from 0, the sender holds without a gap) and then the
observations its neighbours were last heard to lack, oldest first, while the budget lasts.
Observations are relayed, so they cross the team even when two drones are never in range of
each other. Duplicates are harmless: the search map applies each observation id once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerialrange.search.belief import Observation

HEADER_BYTES = 8  # sender id, sequence, time
VECTOR_ENTRY_BYTES = 4
OBS_HEADER_BYTES = 12  # origin, sequence number, time, false alarm intensity
RUN_BYTES = 5  # first cell (3 bytes), run length (1), detection probability level (1)
DETECTION_BYTES = 16  # north, east, two variances, as 32-bit floats


_SIZE: dict[int, tuple[Observation, int]] = {}


def observation_bytes(obs: Observation) -> int:
    """Encoded size, computed once per observation object (the cache holds the object, so its
    identity cannot be reused; cleared at the start of each search by `clear_size_cache`)."""
    hit = _SIZE.get(id(obs))
    if hit is None or hit[0] is not obs:
        hit = (obs, _observation_bytes(obs))
        _SIZE[id(obs)] = hit
    return hit[1]


def clear_size_cache() -> None:
    _SIZE.clear()


def _observation_bytes(obs: Observation) -> int:
    """Encoded size: the cells as runs of consecutive numbers with one detection probability
    (stored as one of 256 levels), plus each detection."""
    cells = np.asarray(obs.cells, np.int64)
    if len(cells) == 0:
        runs = 0
    else:
        order = np.argsort(cells)
        c = cells[order]
        lvl = np.round(np.asarray(obs.pd, float)[order] * 255)
        runs = 1 + int(np.count_nonzero((np.diff(c) != 1) | (np.diff(lvl) != 0)))
    return OBS_HEADER_BYTES + RUN_BYTES * runs + DETECTION_BYTES * len(obs.detections)


@dataclass(frozen=True)
class Radio:
    range_m: float | None = None  # None: every drone in range
    loss: float = 0.0
    budget_bytes: float | None = None  # per drone per second

    def __post_init__(self) -> None:
        if not 0.0 <= self.loss <= 1.0:
            raise ValueError("loss is a probability")


@dataclass
class Node:
    """One drone's store of observations and what it last heard from the others."""

    me: int
    n_drones: int
    store: dict[int, dict[int, Observation]] = field(default_factory=dict)
    heard: dict[int, np.ndarray] = field(default_factory=dict)  # peer -> its version vector
    sent_bytes: int = 0
    credit: float = 0.0
    prefix: dict[int, int] = field(default_factory=dict)  # origin -> gap-free prefix length
    top: dict[int, int] = field(default_factory=dict)  # origin -> highest sequence held

    def add(self, obs: Observation) -> bool:
        s = self.store.setdefault(obs.drone, {})
        if obs.seq in s:
            return False
        s[obs.seq] = obs
        self.top[obs.drone] = max(self.top.get(obs.drone, -1), obs.seq)
        k = self.prefix.get(obs.drone, 0)
        while k in s:
            k += 1
        self.prefix[obs.drone] = k
        return True

    def vector(self) -> np.ndarray:
        """For each origin, the length of the gap-free prefix held."""
        v = np.zeros(self.n_drones, np.int64)
        for o, k in self.prefix.items():
            v[o] = k
        return v

    def packet(self, neighbours: list[int], budget: float | None) -> list[Observation]:
        """Observations to broadcast now: those some current neighbour lacks (by its last heard
        vector; a neighbour never heard from lacks everything), oldest first, within budget."""
        base = HEADER_BYTES + VECTOR_ENTRY_BYTES * self.n_drones
        if budget is not None and budget < base:
            return []
        room = None if budget is None else budget - base
        lack = np.full(self.n_drones, np.iinfo(np.int64).max)
        for p in neighbours:
            v = self.heard.get(p)
            have = v if v is not None else np.zeros(self.n_drones, np.int64)
            lack = np.minimum(lack, have)
        if not neighbours:
            return []
        want = []
        for o, s in self.store.items():
            for seq in range(int(lack[o]), self.top[o] + 1):
                obs = s.get(seq)
                if obs is not None:
                    want.append(obs)
        want.sort(key=lambda x: (x.t_s, x.drone, x.seq))
        out = []
        for obs in want:
            b = observation_bytes(obs)
            if room is not None:
                if b > room:
                    break
                room -= b
            out.append(obs)
        return out


def exchange(
    nodes: list[Node],
    pos_ne: np.ndarray,
    radio: Radio,
    dt_s: float,
    rng: np.random.Generator,
) -> list[list[Observation]]:
    """One broadcast round: every drone sends once. Returns, for each drone, the
    observations it received that it did not hold before (in arrival order)."""
    n = len(nodes)
    d = np.hypot(*(pos_ne[:, None, :] - pos_ne[None, :, :]).transpose(2, 0, 1))
    in_range = np.ones((n, n), bool) if radio.range_m is None else d <= radio.range_m
    np.fill_diagonal(in_range, False)
    new: list[list[Observation]] = [[] for _ in range(n)]
    sends = []
    for i, node in enumerate(nodes):
        budget = None
        if radio.budget_bytes is not None:
            node.credit = min(node.credit + radio.budget_bytes * dt_s, radio.budget_bytes * dt_s)
            budget = node.credit
        nb = [int(j) for j in np.nonzero(in_range[i])[0]]
        obs = node.packet(nb, budget)
        size = HEADER_BYTES + VECTOR_ENTRY_BYTES * n + sum(observation_bytes(o) for o in obs)
        if budget is not None:
            if size > budget:  # not even the vector fits: stay silent this round
                sends.append((i, None, []))
                continue
            node.credit -= size
        node.sent_bytes += size
        sends.append((i, node.vector(), obs))
    for i, vec, obs in sends:
        if vec is None:
            continue
        for j in np.nonzero(in_range[i])[0]:
            if radio.loss > 0 and rng.random() < radio.loss:
                continue
            nodes[j].heard[i] = vec
            for o in obs:
                if nodes[j].add(o):
                    new[j].append(o)
    return new
