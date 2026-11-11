"""Multi-drone search simulation (Phase 14, D-752).

Time advances in steps of `dt_s`. Each drone flies at constant speed and height toward its
next waypoint, looking along its heading; its camera's usable footprint (sensor.py) is
rasterised onto the grid. A cell's pass begins when it enters a drone's footprint and ends
when it leaves; when the pass over the person's cell ends, the person is detected with the
measured probability for a pass of that length (frames in view; detection rises steeply with
it, results/multi_drone/sensor_*.json). False alarms arise over the ground whose passes ended,
at the measured rate per square metre. Detections are emitted as the ROS 2 node's messages and read
back by the drone's search map. Each step's ended passes form one observation, applied to the
drone's own map and shared by radio (comms.py) unless sharing is off.

Planners (all decentralised: each drone decides from its own map):
- `lawnmower`: the area is cut into one strip per drone along east; each drone sweeps its strip
  in parallel lanes, the lane spacing being the footprint's narrowest cross-track width so no
  ground is skipped, and repeats in reverse. Fixed at launch; needs no radio.
- `greedy`: the area is cut into square blocks of `block_lanes` lanes. When a drone finishes a
  block it picks the block with the highest rate of expected detection, the probability mass
  it holds times the detection probability, divided by the time to fly there and sweep it, and
  sweeps it in lanes.
- `auction`: as greedy, and each choice is announced as a claim (block, bid = the rate) over
  the same radio; a drone does not choose a block that another drone has claimed with a higher
  bid (ties to the lower drone number) until that claim's expected end time. A single-item
  auction in the market-based family (Dias, Zlot, Kalra and Stentz, Proc. IEEE 94(7), 2006).

The person is stationary. Turns are instantaneous; drone endurance is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aerialrange.data.syndrone import FPS
from aerialrange.search import comms as cm
from aerialrange.search.belief import Observation, SearchMap, clear_likelihood_cache
from aerialrange.search.grid import Grid
from aerialrange.search.sensor import SensorModel, observation_detections, tracked_object_array

PLANNERS = ("lawnmower", "greedy", "auction")
CLAIM_BYTES = 12  # block (4), bid (4), end time (4)


@dataclass(frozen=True)
class Scenario:
    grid: Grid
    prior: NDArray
    sensor: SensorModel
    n_drones: int
    speed_ms: float
    launch_ne: tuple[float, float]
    horizon_s: float
    dt_s: float = 1.0


@dataclass(frozen=True)
class PlannerConfig:
    kind: str
    share: bool = True
    block_lanes: int = 4

    def __post_init__(self) -> None:
        if self.kind not in PLANNERS:
            raise ValueError(f"planner must be one of {PLANNERS}")


@dataclass
class Result:
    found: bool
    t_found_s: float | None
    false_alarms_before_find: int
    false_alarms_total: int
    cell_passes: int
    repeat_cell_passes: int
    bytes_sent: int
    found_curve: NDArray = field(repr=False)  # per step: 1 once found


def lane_spacing_m(sensor: SensorModel) -> float:
    """Narrowest cross-track width of the usable footprint, flying north."""
    ne = sensor.footprint(0.0, 0.0, 0.0)
    if len(ne) == 0:
        raise ValueError("no usable footprint at this height and tilt")
    near = abs(ne[2, 1] - ne[3, 1])
    far = abs(ne[1, 1] - ne[0, 1])
    return float(min(near, far))


def lanes(
    n_lo: float, n_hi: float, e_lo: float, e_hi: float, spacing: float, start_ne: tuple
) -> list[tuple[float, float]]:
    """Boustrophedon waypoints over a rectangle, lanes along north, spaced `spacing` apart
    with the first lane half a spacing inside the edge; starts from the corner nearest
    `start_ne`."""
    k = max(int(np.ceil((e_hi - e_lo) / spacing)), 1)
    es = e_lo + (np.arange(k) + 0.5) * (e_hi - e_lo) / k
    if abs(es[-1] - start_ne[1]) < abs(es[0] - start_ne[1]):
        es = es[::-1]
    up = abs(n_lo - start_ne[0]) <= abs(n_hi - start_ne[0])
    pts = []
    for e in es:
        a, b = (n_lo, n_hi) if up else (n_hi, n_lo)
        pts += [(a, e), (b, e)]
        up = not up
    return pts


class Drone:
    def __init__(self, i: int, n0: float, e0: float) -> None:
        self.i = i
        self.pos = np.array([n0, e0], float)
        self.yaw = 0.0
        self.route: list[tuple[float, float]] = []
        self.inview: dict[int, int] = {}  # cell -> step its pass began
        self.seq = 0
        self.block: int | None = None

    def step(self, speed: float, dt: float) -> None:
        left = speed * dt
        while left > 1e-9 and self.route:
            tgt = np.asarray(self.route[0], float)
            d = tgt - self.pos
            dist = float(np.hypot(*d))
            if dist < 1e-9:
                self.route.pop(0)
                continue
            self.yaw = float(np.arctan2(d[1], d[0]))
            if dist <= left:
                self.pos = tgt
                self.route.pop(0)
                left -= dist
            else:
                self.pos = self.pos + d / dist * left
                left = 0.0


def block_layout(grid: Grid, block_cells: int) -> tuple[int, int]:
    return int(np.ceil(grid.nrows / block_cells)), int(np.ceil(grid.ncols / block_cells))


def block_mass(p: NDArray, grid: Grid, block_cells: int) -> NDArray:
    br, bc = block_layout(grid, block_cells)
    pad = np.zeros((br * block_cells, bc * block_cells))
    pad[: grid.nrows, : grid.ncols] = p.reshape(grid.nrows, grid.ncols)
    return pad.reshape(br, block_cells, bc, block_cells).sum(axis=(1, 3)).ravel()


def block_bounds(grid: Grid, block_cells: int, b: int) -> tuple[float, float, float, float]:
    _, bc = block_layout(grid, block_cells)
    r, c = divmod(b, bc)
    n_lo = grid.n0_m + r * block_cells * grid.cell_m
    e_lo = grid.e0_m + c * block_cells * grid.cell_m
    n_hi = min(n_lo + block_cells * grid.cell_m, grid.n0_m + grid.nrows * grid.cell_m)
    e_hi = min(e_lo + block_cells * grid.cell_m, grid.e0_m + grid.ncols * grid.cell_m)
    return n_lo, n_hi, e_lo, e_hi


def run(sc: Scenario, pc: PlannerConfig, radio: cm.Radio, seed: int) -> Result:
    """One search. The person's position and every random outcome come from `seed`."""
    from aerialrange.search.prior import sample_location

    cm.clear_size_cache()
    clear_likelihood_cache()
    rng_truth = np.random.default_rng([seed, 0])
    rng_sense = np.random.default_rng([seed, 1])
    rng_radio = np.random.default_rng([seed, 2])
    g, s = sc.grid, sc.sensor
    person = sample_location(sc.prior, g, rng_truth)
    person_cell = int(g.cell_of(person[0], person[1]))
    spacing = lane_spacing_m(s)
    block_cells = max(int(round(pc.block_lanes * spacing / g.cell_m)), 1)
    drones = [Drone(i, *sc.launch_ne) for i in range(sc.n_drones)]
    maps = [SearchMap.from_prior(g, sc.prior) for _ in drones]
    nodes = [cm.Node(i, sc.n_drones) for i in range(sc.n_drones)]
    claims: list[dict[int, tuple[float, int, float]]] = [{} for _ in drones]  # block->(bid,d,end)
    claim_out: list[list[tuple[int, float, int, float]]] = [[] for _ in drones]
    passes_seen = np.zeros(g.size, np.int32)
    n_steps = int(np.ceil(sc.horizon_s / sc.dt_s))
    curve = np.zeros(n_steps, np.uint8)
    found_t, fa_before, fa_total, n_pass, n_repeat = None, 0, 0, 0, 0
    track_id = 1
    for k in range(n_steps):
        t = (k + 1) * sc.dt_s
        for d in drones:
            if not d.route:
                plan(d, pc, sc, maps[d.i], claims[d.i], claim_out[d.i], spacing, block_cells, t)
            d.step(sc.speed_ms, sc.dt_s)
        new_obs: list[Observation | None] = []
        for d in drones:
            fp = s.footprint(float(d.pos[0]), float(d.pos[1]), d.yaw)
            now = set(g.cells_in_polygon(fp).tolist()) if len(fp) else set()
            ended = np.array(sorted(set(d.inview) - now), np.int64)
            frames = np.array([(k - d.inview[c]) for c in ended.tolist()], float) * sc.dt_s * FPS
            d.inview = {c: d.inview.get(c, k) for c in now}
            pd_pass = s.pd_for(frames)
            if len(ended) == 0:
                new_obs.append(None)
                continue
            n_pass += len(ended)
            n_repeat += int(np.count_nonzero(passes_seen[ended]))
            passes_seen[ended] += 1
            dets: list[tuple[float, float]] = []
            true_hit = False
            hit = np.nonzero(ended == person_cell)[0]
            if len(hit):
                if rng_sense.random() < pd_pass[hit[0]]:
                    dets.append(
                        (
                            person[0] + rng_sense.normal(0, s.pos_sigma_m),
                            person[1] + rng_sense.normal(0, s.pos_sigma_m),
                        )
                    )
                    true_hit = True
            n_fa = int(rng_sense.poisson(s.fa_per_m2 * len(ended) * g.cell_area_m2))
            if n_fa:
                cells = rng_sense.choice(ended, n_fa)
                r, c = np.divmod(cells, g.ncols)
                fn = g.n0_m + (r + rng_sense.random(n_fa)) * g.cell_m
                fe = g.e0_m + (c + rng_sense.random(n_fa)) * g.cell_m
                dets += list(zip(fn.tolist(), fe.tolist(), strict=True))
                fa_total += n_fa
                if found_t is None:
                    fa_before += n_fa
            if true_hit and found_t is None:
                found_t = t
            msg = tracked_object_array(t, k + 1, dets, s.pos_sigma_m, track_id)
            track_id += len(dets)
            obs = Observation(
                d.i,
                d.seq,
                t,
                ended,
                pd_pass,
                observation_detections(msg),
                s.fa_per_m2,
            )
            d.seq += 1
            maps[d.i].apply(obs)
            nodes[d.i].add(obs)
            new_obs.append(obs)
        if pc.share and pc.kind != "lawnmower" and sc.n_drones > 1:
            pos = np.array([d.pos for d in drones])
            got = cm.exchange(nodes, pos, radio, sc.dt_s, rng_radio)
            for i, lst in enumerate(got):
                for o in lst:
                    maps[i].apply(o)
            if pc.kind == "auction":
                share_claims(drones, claims, claim_out, pos, radio, rng_radio, nodes)
                for d in drones:  # a better claim heard for my block: choose again
                    if d.block is not None and claims[d.i].get(d.block, (0, d.i))[1] != d.i:
                        d.route, d.block = [], None
        if found_t is not None:
            curve[k:] = 1
            break
    return Result(
        found=found_t is not None,
        t_found_s=found_t,
        false_alarms_before_find=fa_before,
        false_alarms_total=fa_total,
        cell_passes=n_pass,
        repeat_cell_passes=n_repeat,
        bytes_sent=sum(n.sent_bytes for n in nodes),
        found_curve=curve,
    )


def plan(
    d: Drone,
    pc: PlannerConfig,
    sc: Scenario,
    smap: SearchMap,
    claims: dict[int, tuple[float, int, float]],
    out: list[tuple[int, float, int, float]],
    spacing: float,
    block_cells: int,
    t: float,
) -> None:
    g = sc.grid
    if pc.kind == "lawnmower":  # this drone's strip, again and again
        width = g.ncols * g.cell_m / sc.n_drones
        e_lo = g.e0_m + d.i * width
        d.route = lanes(
            g.n0_m, g.n0_m + g.nrows * g.cell_m, e_lo, e_lo + width, spacing, tuple(d.pos)
        )
        return
    p, _ = smap.probabilities()
    mass = block_mass(p, g, block_cells)
    br, bc = block_layout(g, block_cells)
    side = block_cells * g.cell_m
    r, c = np.divmod(np.arange(br * bc), bc)
    cn = g.n0_m + (r + 0.5) * side
    ce = g.e0_m + (c + 0.5) * side
    travel = np.hypot(cn - d.pos[0], ce - d.pos[1]) / sc.speed_ms
    sweep = (side / spacing) * side / sc.speed_ms
    rate = mass * sc.sensor.pd_per_pass / (travel + sweep)
    if pc.kind == "auction":
        for b, (bid, who, end) in list(claims.items()):
            if end < t:
                del claims[b]
            elif who != d.i and (bid > rate[b] or (bid == rate[b] and who < d.i)):
                rate[b] = -1.0
    b = int(np.argmax(rate))
    n_lo, n_hi, e_lo, e_hi = block_bounds(g, block_cells, b)
    d.route = lanes(n_lo, n_hi, e_lo, e_hi, spacing, tuple(d.pos))
    d.block = b
    if pc.kind == "auction":
        end = t + travel[b] + sweep
        claims[b] = (float(rate[b]), d.i, float(end))
        out.append((b, float(rate[b]), d.i, float(end)))


def share_claims(
    drones: list[Drone],
    claims: list[dict[int, tuple[float, int, float]]],
    out: list[list[tuple[int, float, int, float]]],
    pos: NDArray,
    radio: cm.Radio,
    rng: np.random.Generator,
    nodes: list[cm.Node],
) -> None:
    """Broadcast each drone's new claims once over the radio (range and loss apply; their
    bytes are counted against the sender)."""
    n = len(drones)
    dd = np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1))
    reach = np.ones((n, n), bool) if radio.range_m is None else dd <= radio.range_m
    np.fill_diagonal(reach, False)
    for i in range(n):
        if not out[i]:
            continue
        nodes[i].sent_bytes += CLAIM_BYTES * len(out[i])
        for j in np.nonzero(reach[i])[0]:
            if radio.loss > 0 and rng.random() < radio.loss:
                continue
            for b, bid, who, end in out[i]:
                old = claims[j].get(b)
                if old is None or (bid, -who) > (old[0], -old[1]):
                    claims[j][b] = (bid, who, end)
        out[i].clear()
