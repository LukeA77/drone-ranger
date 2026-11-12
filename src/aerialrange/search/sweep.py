"""Sweep first, then look again (Phase 16, D-764): leaderless lanes, and a coordinator option.

Sweep: the area is cut into one strip per drone along east, as the Phase 14 lawnmower, and each
strip into lanes along north (`lane_spacing` apart, the first half a spacing inside the edge).
Every drone computes the same lanes from the team size and the area, so no plan is sent. A lane
flown in sweep round r is a task: a drone flies it from its nearer end to the other and reports
it finished in its log (compact.py, DONE_BIT).

Choosing the next lane, from what this drone knows (its own log and the logs it has received),
in the earliest round that still has lanes neither finished nor being flown by another live
drone:
1. the first such lane of its own strip, in sweep order;
2. otherwise, work sharing: from the strip with the most such lanes (ties: lower strip), the
   last one in sweep order, the end farthest from that strip's owner.
After the sweep (`post`): "resweep" starts the next round of lanes; "map" chooses blocks for
second looks as the Phase 14 auction does (sim.plan's expected detection rate), skipping blocks
another live drone is on; "coverage" does the same on a map built only from where the camera
looked and saw nothing (detections left out, so look-alikes do not pull the drones back).
Two drones on the same task: the one with the higher number chooses again. A drone is presumed
lost when it was heard from before and nothing newer than `lost_after_s` has arrived; its
unfinished lanes are then free. A drone never heard from is not presumed lost.

Coordinator (option): drone 0 also computes, each step, a next task for every drone it knows of,
by an optimal assignment (Hungarian method, scipy.optimize.linear_sum_assignment) of drones to
free tasks: lanes of the current round by flight time from where each drone's current task
ends; once no round is left, blocks by the expected detection rate. The table travels in the
coordinator's log. A drone that finishes a task takes its entry in the latest table it holds, if
that table is no older than `lost_after_s` and the task is still free by its own knowledge;
otherwise it chooses as above. Losing the coordinator therefore falls back to the leaderless
rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from aerialrange.search import sim
from aerialrange.search.compact import DONE_BIT, NO_TASK, CompactNode
from aerialrange.search.grid import Grid


@dataclass(frozen=True)
class Lane:
    strip: int
    order: int  # position in the strip's sweep order
    e_m: float
    n_lo: float
    n_hi: float


MAX_ROUNDS = 8  # sweep rounds with task numbers; the horizon is three sweeps' time


@dataclass(frozen=True)
class Layout:
    lanes: tuple[Lane, ...]
    n_strips: int
    block_cells: int
    n_blocks: int
    rounds: int = MAX_ROUNDS

    def lane_task(self, i: int, rnd: int = 0) -> int:
        return rnd * len(self.lanes) + i

    def lane_of(self, task: int) -> int:
        return task % len(self.lanes)

    def round_of(self, task: int) -> int:
        return task // len(self.lanes)

    def block_task(self, b: int) -> int:
        return self.rounds * len(self.lanes) + b

    def block_of(self, task: int) -> int:
        return task - self.rounds * len(self.lanes)

    def is_lane(self, task: int) -> bool:
        return 0 <= task < self.rounds * len(self.lanes)

    def strip_lanes(self, s: int) -> list[int]:
        return sorted(
            (i for i, ln in enumerate(self.lanes) if ln.strip == s),
            key=lambda i: self.lanes[i].order,
        )


def layout(grid: Grid, n_drones: int, spacing: float, launch_ne: tuple, block_cells: int) -> Layout:
    """Lanes of every strip, each strip's sweep order starting from the side nearer the
    launch point (as sim.lanes starts from the nearer corner)."""
    width = grid.ncols * grid.cell_m / n_drones
    n_lo, n_hi = grid.n0_m, grid.n0_m + grid.nrows * grid.cell_m
    out = []
    for s in range(n_drones):
        e_lo = grid.e0_m + s * width
        k = max(int(np.ceil(width / spacing)), 1)
        es = e_lo + (np.arange(k) + 0.5) * width / k
        if abs(es[-1] - launch_ne[1]) < abs(es[0] - launch_ne[1]):
            es = es[::-1]
        out += [Lane(s, j, float(e), n_lo, n_hi) for j, e in enumerate(es)]
    br, bc = sim.block_layout(grid, block_cells)
    if MAX_ROUNDS * len(out) + br * bc >= NO_TASK:
        raise ValueError("too many tasks for the 15-bit task field")
    return Layout(tuple(out), n_drones, block_cells, br * bc)


def lane_route(lane: Lane, pos: np.ndarray) -> list[tuple[float, float]]:
    a, b = (lane.n_lo, lane.n_hi)
    if abs(pos[0] - b) < abs(pos[0] - a):
        a, b = b, a
    return [(a, lane.e_m), (b, lane.e_m)]


def live(node: CompactNode, t: float, dt: float, lost_after_s: float, me: int) -> dict[int, int]:
    """Other drones not presumed lost -> the task in their newest entry (NO_TASK if none)."""
    out = {}
    for o in range(node.n_drones):
        if o == me:
            continue
        e = node.latest.get(o)
        if e is None:
            out[o] = NO_TASK  # never heard: not presumed lost, task unknown
            continue
        if t - (e.step + 1) * dt > lost_after_s:
            continue
        out[o] = e.task & ~DONE_BIT if not (e.task & DONE_BIT) else NO_TASK
    return out


def free_lanes(lay: Layout, node: CompactNode, busy: dict[int, int], rnd: int = 0) -> set[int]:
    """Lanes of round `rnd` neither reported finished nor being flown by a live drone."""
    taken = set(busy.values())
    return {
        i
        for i in range(len(lay.lanes))
        if lay.lane_task(i, rnd) not in node.done and lay.lane_task(i, rnd) not in taken
    }


def current_round(
    lay: Layout, node: CompactNode, busy: dict[int, int], max_rounds: int
) -> tuple[int, set[int]] | None:
    """(earliest round with free lanes, its free lanes), or None when no round is left."""
    for r in range(min(max_rounds, lay.rounds)):
        f = free_lanes(lay, node, busy, r)
        if f:
            return r, f
    return None


def choose_lane(lay: Layout, me: int, free: set[int]) -> int | None:
    own = [i for i in lay.strip_lanes(me % lay.n_strips) if i in free]
    if own:
        return own[0]
    best, best_n = None, 0
    for s in range(lay.n_strips):
        f = [i for i in lay.strip_lanes(s) if i in free]
        if len(f) > best_n:
            best, best_n = f[-1], len(f)
    return best


def block_rates(
    p: np.ndarray,
    grid: Grid,
    lay: Layout,
    pos: np.ndarray,
    speed: float,
    spacing: float,
    pd_per_pass: float,
) -> np.ndarray:
    """sim.plan's expected detection rate of every block from `pos`."""
    mass = sim.block_mass(p, grid, lay.block_cells)
    br, bc = sim.block_layout(grid, lay.block_cells)
    side = lay.block_cells * grid.cell_m
    r, c = np.divmod(np.arange(br * bc), bc)
    cn = grid.n0_m + (r + 0.5) * side
    ce = grid.e0_m + (c + 0.5) * side
    travel = np.hypot(cn - pos[0], ce - pos[1]) / speed
    sweep_s = (side / spacing) * side / speed
    return mass * pd_per_pass / (travel + sweep_s)


def task_end(lay: Layout, task: int, pos: np.ndarray, grid: Grid) -> np.ndarray:
    """Where a drone at `pos` doing `task` will be when it finishes (lanes: the far end;
    blocks and unknown: where it is, an approximation used only by the coordinator)."""
    if lay.is_lane(task):
        r = lane_route(lay.lanes[lay.lane_of(task)], pos)
        return np.asarray(r[-1], float)
    return np.asarray(pos, float)


def coordinator_table(
    lay: Layout,
    node: CompactNode,
    me_pos: np.ndarray,
    me_task: int,
    t: float,
    dt: float,
    lost_after_s: float,
    p: np.ndarray,
    grid: Grid,
    speed: float,
    spacing: float,
    pd_per_pass: float,
    max_rounds: int,
) -> tuple[int, ...]:
    """Next task for every drone (NO_TASK where unknown), by optimal assignment."""
    n = node.n_drones
    who, ends, cur = [], [], {}
    for o in range(n):
        if o == node.me:
            who.append(o)
            ends.append(task_end(lay, me_task, me_pos, grid))
            cur[o] = me_task
            continue
        e = node.latest.get(o)
        if e is None or t - (e.step + 1) * dt > lost_after_s:
            continue
        en, ee, _ = e.pose()
        task = NO_TASK if e.task & DONE_BIT else e.task
        who.append(o)
        ends.append(task_end(lay, task, np.array([en, ee]), grid))
        cur[o] = task
    table = [NO_TASK] * n
    ends_a = np.array(ends)
    cr = current_round(lay, node, dict(cur), max_rounds)
    if cr is not None:
        rnd, free = cr
        lanes = sorted(free)
        cost = np.zeros((len(who), len(lanes)))
        for j, i in enumerate(lanes):
            ln = lay.lanes[i]
            d_lo = np.hypot(ends_a[:, 0] - ln.n_lo, ends_a[:, 1] - ln.e_m)
            d_hi = np.hypot(ends_a[:, 0] - ln.n_hi, ends_a[:, 1] - ln.e_m)
            cost[:, j] = np.minimum(d_lo, d_hi) / speed
        rows, cols = linear_sum_assignment(cost)
        for r_, c_ in zip(rows, cols, strict=True):
            table[who[r_]] = lay.lane_task(lanes[c_], rnd)
        return tuple(table)
    taken = {lay.block_of(k) for k in cur.values() if k != NO_TASK and not lay.is_lane(k)}
    rates = np.array(
        [block_rates(p, grid, lay, en_, speed, spacing, pd_per_pass) for en_ in ends_a]
    )
    if taken:
        rates[:, sorted(taken)] = -1.0
    rows, cols = linear_sum_assignment(-rates)
    for r_, c_ in zip(rows, cols, strict=True):
        if rates[r_, c_] > 0:
            table[who[r_]] = lay.block_task(int(c_))
    return tuple(table)
