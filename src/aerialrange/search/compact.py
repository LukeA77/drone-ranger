"""Compact observation sharing (Phase 16, D-764): send the flight, not the cells.

Phase 14 (comms.py) sent each observation as the cells it covered, about 100 bytes, and every
observation was broadcast about 13 times across a team that could all hear each other
(results/multi_drone/bytes_p10_v2_cal.json). Here:

- **Pose log.** Each step a drone logs one entry: its pose quantised to 0.1 m (north, east) and
  1/65536 of a turn (heading) in 16-bit integers, the task it is on (16 bits, the top bit
  marking a task finished this step), and its person detections as 0.1 m offsets from that pose.
  Every drone, the sender included, rebuilds the observations by replaying each sender's log in
  order through the same footprint and pass code (`Replayer`), so all maps are built from
  identical observations. The camera model (footprint, detection probability by pass length,
  false-alarm rate, position error) is the team's shared configuration, so it is not sent.
- **No echo.** In each round drones transmit one after another in a random order. A drone does
  not send an entry it has heard transmitted (by anyone, itself included) in the last
  `SUPPRESS_ROUNDS` rounds: a neighbour that heard it says so in its next summary, which arrives
  within a round, and the second round covers the transmission order. After that, an entry is
  sent again only if a neighbour's latest summary still lacks it, so losses are repaired.
- **Summary.** Each packet carries, for each drone, how many of its entries the sender holds
  without a gap (as comms.py's version vector), in 16 bits.
- **Budget.** As the D-756 follow-up: the allowance builds up until the next message fits, so
  the average rate keeps to the budget and nothing is blocked for ever. Entries go oldest
  first; a replay needs each sender's history in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerialrange.data.syndrone import FPS
from aerialrange.search.belief import Detection, Observation
from aerialrange.search.comms import Radio
from aerialrange.search.grid import Grid
from aerialrange.search.sensor import SensorModel

POS_Q_M = 0.1  # pose and detection quantum (16-bit signed: +-3.2 km, the area is 2.2 km)
YAW_Q = 2 * np.pi / 65536
ENTRY_BYTES = 8  # north, east, heading, task (16 bits each)
DET_COUNT_BYTES = 1
DET_BYTES = 4  # north and east offsets from the pose
PACKET_HEADER_BYTES = 3  # sender, round
SUMMARY_ENTRY_BYTES = 2  # per drone
GROUP_BYTES = 4  # origin (1), first sequence number (2), count (1) per run of entries
TABLE_ENTRY_BYTES = 2  # coordinator's next task per drone
SUPPRESS_ROUNDS = 2
DONE_BIT = 1 << 15
NO_TASK = DONE_BIT - 1


def quantise(n_m: float, e_m: float, yaw_rad: float) -> tuple[int, int, int]:
    qn, qe = int(round(n_m / POS_Q_M)), int(round(e_m / POS_Q_M))
    if not (-32768 <= qn < 32768 and -32768 <= qe < 32768):
        raise ValueError("pose outside the 16-bit range")
    return qn, qe, int(round((yaw_rad % (2 * np.pi)) / YAW_Q)) % 65536


def dequantise(qn: int, qe: int, qyaw: int) -> tuple[float, float, float]:
    return qn * POS_Q_M, qe * POS_Q_M, qyaw * YAW_Q


@dataclass(frozen=True)
class Entry:
    """One step of one drone's flight log."""

    origin: int
    seq: int  # step number of this drone's log, from 0
    step: int  # simulation step (k)
    qn: int
    qe: int
    qyaw: int
    task: int = NO_TASK  # task id, DONE_BIT set when it was finished this step
    dets: tuple[tuple[int, int], ...] = ()  # offsets from the pose, POS_Q_M units
    table: tuple[int, ...] | None = None  # coordinator only: next task for each drone

    @property
    def id(self) -> tuple[int, int]:
        return (self.origin, self.seq)

    @property
    def nbytes(self) -> int:
        b = ENTRY_BYTES + DET_COUNT_BYTES + DET_BYTES * len(self.dets)
        return b + (TABLE_ENTRY_BYTES * len(self.table) if self.table is not None else 0)

    def pose(self) -> tuple[float, float, float]:
        return dequantise(self.qn, self.qe, self.qyaw)

    def detections_ne(self) -> list[tuple[float, float]]:
        n, e, _ = self.pose()
        return [(n + a * POS_Q_M, e + b * POS_Q_M) for a, b in self.dets]


def make_entry(
    origin: int,
    seq: int,
    step: int,
    pos_ne: tuple[float, float],
    yaw: float,
    dets_ne: list[tuple[float, float]],
    task: int = NO_TASK,
    table: tuple[int, ...] | None = None,
) -> Entry:
    qn, qe, qy = quantise(pos_ne[0], pos_ne[1], yaw)
    n, e, _ = dequantise(qn, qe, qy)
    off = []
    for dn, de in dets_ne:
        a, b = int(round((dn - n) / POS_Q_M)), int(round((de - e) / POS_Q_M))
        if not (-32768 <= a < 32768 and -32768 <= b < 32768):
            raise ValueError("detection too far from the pose")
        off.append((a, b))
    return Entry(origin, seq, step, qn, qe, qy, task, tuple(off), table)


@dataclass
class Replayer:
    """Rebuilds one drone's observations from its log, in order (same code for every drone)."""

    grid: Grid
    sensor: SensorModel
    dt_s: float
    inview: dict[int, int] = field(default_factory=dict)
    next_seq: int = 0

    def apply(self, e: Entry) -> Observation | None:
        if e.seq != self.next_seq:
            raise ValueError("entries must be replayed in order")
        self.next_seq += 1
        n, m, yaw = e.pose()
        fp = self.sensor.footprint(n, m, yaw)
        now = set(self.grid.cells_in_polygon(fp).tolist()) if len(fp) else set()
        ended = np.array(sorted(set(self.inview) - now), np.int64)
        frames = np.array([(e.step - self.inview[c]) for c in ended.tolist()], float)
        frames = frames * self.dt_s * FPS
        self.inview = {c: self.inview.get(c, e.step) for c in now}
        if len(ended) == 0 and not e.dets:
            return None
        s = self.sensor
        cov = (s.pos_sigma_m**2, 0.0, 0.0, s.pos_sigma_m**2)
        dets = tuple(Detection(a, b, cov) for a, b in e.detections_ne())
        t = (e.step + 1) * self.dt_s
        return Observation(e.origin, e.seq, t, ended, s.pd_for(frames), dets, s.fa_per_m2)


@dataclass
class CompactNode:
    me: int
    n_drones: int
    store: dict[int, dict[int, Entry]] = field(default_factory=dict)
    prefix: np.ndarray = field(default=None)  # type: ignore[assignment]
    heard: dict[int, np.ndarray] = field(default_factory=dict)  # neighbour -> its summary
    last_tx: dict[tuple[int, int], int] = field(default_factory=dict)  # entry -> round heard
    sent_bytes: int = 0
    credit: float = 0.0
    latest: dict[int, Entry] = field(default_factory=dict)  # origin -> newest entry held
    done: set[int] = field(default_factory=set)  # tasks reported finished

    def __post_init__(self) -> None:
        if self.prefix is None:
            self.prefix = np.zeros(self.n_drones, np.int64)

    def add(self, e: Entry) -> bool:
        s = self.store.setdefault(e.origin, {})
        if e.seq in s:
            return False
        s[e.seq] = e
        k = int(self.prefix[e.origin])
        while k in s:
            k += 1
        self.prefix[e.origin] = k
        old = self.latest.get(e.origin)
        if old is None or e.seq > old.seq:
            self.latest[e.origin] = e
        if e.task & DONE_BIT:
            self.done.add(e.task & ~DONE_BIT)
        return True

    def wanted(self, neighbours: list[int], rnd: int) -> list[Entry]:
        """Entries some neighbour lacks (by its latest summary; never heard: lacks all) that
        were not heard transmitted in the last SUPPRESS_ROUNDS rounds, oldest first."""
        if not neighbours:
            return []
        lack = np.full(self.n_drones, np.iinfo(np.int64).max)
        for p in neighbours:
            v = self.heard.get(p)
            lack = np.minimum(lack, v if v is not None else 0)
        out = []
        for o, s in self.store.items():
            top = self.latest[o].seq
            for q in range(int(lack[o]), top + 1):
                e = s.get(q)
                if e is None:
                    continue
                last = self.last_tx.get(e.id)
                if last is not None and rnd - last <= SUPPRESS_ROUNDS:
                    continue
                out.append(e)
        out.sort(key=lambda x: (x.step, x.origin, x.seq))
        return out


def packet_bytes(n_drones: int, entries: list[Entry]) -> int:
    groups = 0
    last: tuple[int, int] | None = None
    for e in sorted(entries, key=lambda x: (x.origin, x.seq)):
        if last is None or e.origin != last[0] or e.seq != last[1] + 1:
            groups += 1
        last = (e.origin, e.seq)
    return (
        PACKET_HEADER_BYTES
        + SUMMARY_ENTRY_BYTES * n_drones
        + GROUP_BYTES * groups
        + sum(e.nbytes for e in entries)
    )


def exchange(
    nodes: list[CompactNode],
    alive: list[bool],
    pos_ne: np.ndarray,
    radio: Radio,
    rnd: int,
    dt_s: float,
    rng: np.random.Generator,
) -> list[list[Entry]]:
    """One round: every live drone transmits once, in a random order, and later transmitters
    have heard the earlier ones. Returns, per drone, the entries it received that it did not
    hold before."""
    n = len(nodes)
    d = np.hypot(*(pos_ne[:, None, :] - pos_ne[None, :, :]).transpose(2, 0, 1))
    in_range = np.ones((n, n), bool) if radio.range_m is None else d <= radio.range_m
    np.fill_diagonal(in_range, False)
    ok = np.asarray(alive, bool)
    in_range &= ok[None, :] & ok[:, None]
    new: list[list[Entry]] = [[] for _ in range(n)]
    for i in rng.permutation(n).tolist():
        if not ok[i]:
            continue
        node = nodes[i]
        nb = [int(j) for j in np.nonzero(in_range[i])[0]]
        want = node.wanted(nb, rnd)
        send: list[Entry] = []
        if radio.budget_bytes is None:
            send = want
        else:
            base = packet_bytes(n, [])
            # allowance builds until the next message fits (D-756), capped there
            nxt = packet_bytes(n, want[:1]) if want else base
            cap = max(radio.budget_bytes * dt_s, nxt)
            node.credit = min(node.credit + radio.budget_bytes * dt_s, cap)
            if node.credit < base:
                continue  # silent this round
            for e in want:
                if packet_bytes(n, [*send, e]) > node.credit:
                    break
                send.append(e)
        size = packet_bytes(n, send)
        if radio.budget_bytes is not None:
            node.credit -= size
        node.sent_bytes += size
        summary = node.prefix.copy()
        for e in send:
            node.last_tx[e.id] = rnd
        for j in nb:
            if radio.loss > 0 and rng.random() < radio.loss:
                continue
            r = nodes[j]
            r.heard[i] = summary
            for e in send:
                r.last_tx[e.id] = rnd
                if r.add(e):
                    new[j].append(e)
    return new
