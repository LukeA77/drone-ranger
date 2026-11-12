"""Phase 16 search simulation (D-764): confirmed finds, recurring false alarms, the sweep
planner, compact sharing, a coordinator option and drone loss.

The loop is Phase 14's (sim.run): the same steps, flight, footprint, passes, detections and
random streams. With Phase 14's settings (a Phase 14 planner, frozen sharing, no recurrence, no
loss, stopping at the first detection) it gives exactly sim.run's result for every seed
(tests/test_search_engine.py), so baselines and the new planner are compared in one simulation
under one metric. New random draws (recurring false alarms) use their own stream, so they never
shift Phase 14's draws.

Per step: plan (drones without a route), fly, sense, raise alarms, share. Stopping: at the first
detection of the person (`stop="first"`, Phase 14), or at the confirmed find (`stop="confirmed"`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from aerialrange.data.syndrone import FPS
from aerialrange.search import comms as cm
from aerialrange.search import compact as cp
from aerialrange.search import sim
from aerialrange.search import sweep as sw
from aerialrange.search.belief import Observation, SearchMap, clear_likelihood_cache
from aerialrange.search.confirm import Alarms, ClutterSites
from aerialrange.search.sensor import observation_detections, tracked_object_array

PLANNERS = (*sim.PLANNERS, "sweep")


@dataclass(frozen=True)
class Plan:
    kind: str
    share: bool = True
    block_lanes: int = 4
    sharing: str = "frozen"  # "frozen" (comms.py) or "compact" (compact.py)
    spacing_factor: float = 1.0  # sweep lanes: this times the footprint's narrowest width
    lost_after_s: float = 30.0
    coordinator: bool = False
    post: str = "resweep"  # sweep planner after a sweep: "resweep", "map" or "coverage"

    def __post_init__(self) -> None:
        if self.kind not in PLANNERS:
            raise ValueError(f"planner must be one of {PLANNERS}")
        if self.sharing not in ("frozen", "compact"):
            raise ValueError("sharing is 'frozen' or 'compact'")
        if self.kind == "sweep" and self.sharing != "compact":
            raise ValueError("the sweep planner reads progress from compact logs")
        if self.coordinator and self.kind != "sweep":
            raise ValueError("the coordinator option is for the sweep planner")
        if self.post not in ("resweep", "map", "coverage"):
            raise ValueError("post is 'resweep', 'map' or 'coverage'")


@dataclass(frozen=True)
class World:
    """What Phase 16 adds to the world. Defaults are Phase 14's world."""

    persistent_share: float = 0.0  # share of false alarms from persistent look-alikes
    persistent_fire: float = 0.0  # a look-alike's chance of firing on each pass over it
    confirm_k: int = 2
    confirm_radius_m: float = 0.0  # 0: 5 x the camera's position error (belief.TAIL_SIGMAS)
    stop: str = "first"  # "first" or "confirmed"
    lose: tuple[int, float] | None = None  # (drone, time s): that drone stops for good
    # measured detection probability after a detected and after a missed pass, as multiples
    # of the pass's own probability (None: passes independent, as Phase 14)
    person_memory: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.stop not in ("first", "confirmed"):
            raise ValueError("stop is 'first' or 'confirmed'")
        if not (0 <= self.persistent_fire <= 1 and 0 <= self.persistent_share < 1):
            raise ValueError("look-alike values are probabilities")
        if self.persistent_share > 0 and self.persistent_fire <= 0:
            raise ValueError("persistent look-alikes need a chance of firing")


@dataclass
class Result:
    found: bool  # first detection of the person
    t_found_s: float | None
    confirmed: bool
    t_confirmed_s: float | None
    false_alarms_before_find: int
    false_alarms_total: int
    crew_alarms_before_confirmed: int  # false alarms raised to the crew before the find
    crew_alarms_total: int
    cell_passes: int
    repeat_cell_passes: int
    bytes_sent: int
    steps: int
    found_curve: np.ndarray = field(repr=False)


def run(
    sc: sim.Scenario, pc: Plan, radio: cm.Radio, seed: int, world: World | None = None
) -> Result:
    from aerialrange.search.belief import TAIL_SIGMAS
    from aerialrange.search.prior import sample_location

    world = World() if world is None else world

    cm.clear_size_cache()
    clear_likelihood_cache()
    rng_truth = np.random.default_rng([seed, 0])
    rng_sense = np.random.default_rng([seed, 1])
    rng_radio = np.random.default_rng([seed, 2])
    rng_clutter = np.random.default_rng([seed, 3])
    g, s = sc.grid, sc.sensor
    person = sample_location(sc.prior, g, rng_truth)
    person_cell = int(g.cell_of(person[0], person[1]))
    spacing = sim.lane_spacing_m(s)
    block_cells = max(int(round(pc.block_lanes * spacing / g.cell_m)), 1)
    n = sc.n_drones
    drones = [sim.Drone(i, *sc.launch_ne) for i in range(n)]
    maps = [SearchMap.from_prior(g, sc.prior) for _ in drones]
    compact = pc.sharing == "compact"
    nodes = [cm.Node(i, n) for i in range(n)]
    cnodes = [cp.CompactNode(i, n) for i in range(n)]
    replay = [cp.Replayer(g, s, sc.dt_s) for _ in range(n)]
    obs_of: dict[tuple[int, int], Observation | None] = {}
    applied = [np.zeros(n, np.int64) for _ in range(n)]  # per drone: entries applied per origin
    log_seq = [0] * n
    task = [cp.NO_TASK] * n
    claims: list[dict[int, tuple[float, int, float]]] = [{} for _ in drones]
    claim_out: list[list[tuple[int, float, int, float]]] = [[] for _ in drones]
    lay = None
    if pc.kind == "sweep":
        lay = sw.layout(g, n, spacing * pc.spacing_factor, sc.launch_ne, block_cells)
    alive = [True] * n
    radius = world.confirm_radius_m or TAIL_SIGMAS * s.pos_sigma_m
    alarms = Alarms(world.confirm_k, radius, g.cell_m)
    sites = ClutterSites(g, world.persistent_fire)
    if world.persistent_share > 0:
        sites.place(s.fa_per_m2 * world.persistent_share / world.persistent_fire, rng_clutter)
    fa_rate = s.fa_per_m2 * (1.0 - world.persistent_share)
    last_person: bool | None = None
    passes_seen = np.zeros(g.size, np.int32)
    n_steps = int(np.ceil(sc.horizon_s / sc.dt_s))
    curve = np.zeros(n_steps, np.uint8)
    found_t, conf_t = None, None
    fa_before, fa_total, n_pass, n_repeat, crew_before, crew_total = 0, 0, 0, 0, 0, 0
    track_id = 1
    k = 0
    for k in range(n_steps):
        t = (k + 1) * sc.dt_s
        if world.lose is not None and t > world.lose[1]:
            alive[world.lose[0]] = False
        for d in drones:
            if not alive[d.i]:
                continue
            if not d.route:
                if pc.kind == "sweep":
                    plan_sweep(d, pc, sc, lay, cnodes[d.i], maps[d.i], task, t, spacing)
                else:
                    sim.plan(
                        d,
                        sim.PlannerConfig(pc.kind, pc.share, pc.block_lanes),
                        sc,
                        maps[d.i],
                        claims[d.i],
                        claim_out[d.i],
                        spacing,
                        block_cells,
                        t,
                    )
                    task[d.i] = cp.NO_TASK if d.block is None else lay_block(d.block)
            d.step(sc.speed_ms, sc.dt_s)
        for d in drones:
            if not alive[d.i]:
                continue
            fp = s.footprint(float(d.pos[0]), float(d.pos[1]), d.yaw)
            now = set(g.cells_in_polygon(fp).tolist()) if len(fp) else set()
            ended = np.array(sorted(set(d.inview) - now), np.int64)
            frames = np.array([(k - d.inview[c]) for c in ended.tolist()], float) * sc.dt_s * FPS
            d.inview = {c: d.inview.get(c, k) for c in now}
            pd_pass = s.pd_for(frames)
            finished = pc.kind == "sweep" and not d.route and task[d.i] != cp.NO_TASK
            if len(ended) == 0:
                if compact:
                    _log(
                        d,
                        cnodes,
                        replay,
                        obs_of,
                        log_seq,
                        task,
                        k,
                        [],
                        finished,
                        pc,
                        t,
                        sc,
                        maps,
                        lay,
                        spacing,
                    )
                continue
            n_pass += len(ended)
            n_repeat += int(np.count_nonzero(passes_seen[ended]))
            passes_seen[ended] += 1
            dets: list[tuple[float, float]] = []
            truth: list[bool] = []
            hit = np.nonzero(ended == person_cell)[0]
            if len(hit):
                p_hit = pd_pass[hit[0]]
                if world.person_memory is not None and last_person is not None:
                    p_hit = min(p_hit * world.person_memory[0 if last_person else 1], 1.0)
                seen = rng_sense.random() < p_hit
                last_person = bool(seen)
                if seen:
                    dets.append(
                        (
                            person[0] + rng_sense.normal(0, s.pos_sigma_m),
                            person[1] + rng_sense.normal(0, s.pos_sigma_m),
                        )
                    )
                    truth.append(True)
            n_fa = int(rng_sense.poisson(fa_rate * len(ended) * g.cell_area_m2))
            if n_fa:
                cells = rng_sense.choice(ended, n_fa)
                r, c = np.divmod(cells, g.ncols)
                fn = g.n0_m + (r + rng_sense.random(n_fa)) * g.cell_m
                fe = g.e0_m + (c + rng_sense.random(n_fa)) * g.cell_m
                dets += list(zip(fn.tolist(), fe.tolist(), strict=True))
                truth += [False] * n_fa
            rep = sites.fire(ended, s.pos_sigma_m, rng_clutter)
            dets += rep
            truth += [False] * len(rep)
            n_false = n_fa + len(rep)
            fa_total += n_false
            if found_t is None:
                fa_before += n_false
            if any(truth) and found_t is None:
                found_t = t
            for a, b in dets:
                hit_alarm = alarms.add(a, b, t, (float(person[0]), float(person[1])))
                if hit_alarm is None:
                    continue
                if hit_alarm and conf_t is None:
                    conf_t = t
                elif not hit_alarm:
                    crew_total += 1
                    if conf_t is None:
                        crew_before += 1
            if compact:
                _log(
                    d,
                    cnodes,
                    replay,
                    obs_of,
                    log_seq,
                    task,
                    k,
                    dets,
                    finished,
                    pc,
                    t,
                    sc,
                    maps,
                    lay,
                    spacing,
                )
                continue
            msg = tracked_object_array(t, k + 1, dets, s.pos_sigma_m, track_id)
            track_id += len(dets)
            obs = Observation(
                d.i, d.seq, t, ended, pd_pass, observation_detections(msg), s.fa_per_m2
            )
            d.seq += 1
            maps[d.i].apply(obs)
            nodes[d.i].add(obs)
        if pc.share and pc.kind != "lawnmower" and n > 1:
            pos = np.array([d.pos for d in drones])
            if compact:
                cp.exchange(cnodes, alive, pos, radio, k, sc.dt_s, rng_radio)
                for i in range(n):
                    _apply_known(i, cnodes[i], obs_of, applied, maps)
            else:
                got_f = cm.exchange(nodes, pos, radio, sc.dt_s, rng_radio)
                for i, lst in enumerate(got_f):
                    for o in lst:
                        maps[i].apply(o)
            if pc.kind == "auction":
                sim.share_claims(drones, claims, claim_out, pos, radio, rng_radio, nodes)
                for d in drones:
                    if d.block is not None and claims[d.i].get(d.block, (0, d.i))[1] != d.i:
                        d.route, d.block = [], None
            if pc.kind == "sweep":
                _resolve_conflicts(drones, cnodes, task, alive, t, sc.dt_s, pc.lost_after_s)
        elif compact:
            for i in range(n):
                _apply_known(i, cnodes[i], obs_of, applied, maps)
        done = conf_t is not None if world.stop == "confirmed" else found_t is not None
        if done:
            curve[k:] = 1
            break
    bytes_sent = sum(x.sent_bytes for x in nodes)
    if compact:
        bytes_sent += sum(x.sent_bytes for x in cnodes)
    return Result(
        found=found_t is not None,
        t_found_s=found_t,
        confirmed=conf_t is not None,
        t_confirmed_s=conf_t,
        false_alarms_before_find=fa_before,
        false_alarms_total=fa_total,
        crew_alarms_before_confirmed=crew_before,
        crew_alarms_total=crew_total,
        cell_passes=n_pass,
        repeat_cell_passes=n_repeat,
        bytes_sent=int(bytes_sent),
        steps=k + 1,
        found_curve=curve,
    )


def lay_block(block: int) -> int:
    """Task id for a Phase 14 block (the auction's own number, kept clear of NO_TASK)."""
    return int(block) if block < cp.NO_TASK else cp.NO_TASK


def _log(
    d: sim.Drone,
    cnodes: list[cp.CompactNode],
    replay: list[cp.Replayer],
    obs_of: dict[tuple[int, int], Observation | None],
    log_seq: list[int],
    task: list[int],
    k: int,
    dets: list[tuple[float, float]],
    finished: bool,
    pc: Plan,
    t: float,
    sc: sim.Scenario,
    maps: list[SearchMap],
    lay: sw.Layout | None,
    spacing: float,
) -> None:
    """Append this step's entry to drone d's log, replay it for the shared observation cache,
    and apply it to d's own map (the origin applies its own entries in order at once)."""
    tk = task[d.i]
    code = (tk | cp.DONE_BIT) if finished and tk != cp.NO_TASK else tk
    table = None
    if pc.coordinator and d.i == 0 and lay is not None:
        p, _ = maps[d.i].probabilities()
        table = sw.coordinator_table(
            lay,
            cnodes[d.i],
            d.pos,
            tk,
            t,
            sc.dt_s,
            pc.lost_after_s,
            p,
            sc.grid,
            sc.speed_ms,
            spacing,
            sc.sensor.pd_per_pass,
            max_rounds(pc),
        )
    e = cp.make_entry(
        d.i, log_seq[d.i], k, (float(d.pos[0]), float(d.pos[1])), d.yaw, dets, code, table
    )
    log_seq[d.i] += 1
    ob = replay[d.i].apply(e)
    if ob is not None and pc.post == "coverage" and pc.kind == "sweep":
        ob = dataclasses.replace(ob, detections=())  # the map keeps only where nothing was seen
    obs_of[e.id] = ob
    cnodes[d.i].add(e)
    if finished:
        task[d.i] = cp.NO_TASK


def _apply_known(
    i: int,
    node: cp.CompactNode,
    obs_of: dict[tuple[int, int], Observation | None],
    applied: list[np.ndarray],
    maps: list[SearchMap],
) -> None:
    """Apply, in order, every entry drone i now holds without a gap."""
    for o in range(node.n_drones):
        while applied[i][o] < node.prefix[o]:
            ob = obs_of.get((o, int(applied[i][o])))
            if ob is not None:
                maps[i].apply(ob)
            applied[i][o] += 1


def _resolve_conflicts(
    drones: list[sim.Drone],
    cnodes: list[cp.CompactNode],
    task: list[int],
    alive: list[bool],
    t: float,
    dt: float,
    lost_after_s: float,
) -> None:
    """Two live drones on one task: the higher-numbered one chooses again."""
    for d in drones:
        if not alive[d.i] or task[d.i] == cp.NO_TASK:
            continue
        others = sw.live(cnodes[d.i], t, dt, lost_after_s, d.i)
        if any(o < d.i and tk == task[d.i] for o, tk in others.items()):
            d.route, task[d.i] = [], cp.NO_TASK


def max_rounds(pc: Plan) -> int:
    return sw.MAX_ROUNDS if pc.post == "resweep" else 1


def plan_sweep(
    d: sim.Drone,
    pc: Plan,
    sc: sim.Scenario,
    lay: sw.Layout,
    node: cp.CompactNode,
    smap: SearchMap,
    task: list[int],
    t: float,
    spacing: float,
) -> None:
    """Next task for drone d by the sweep rules (sweep.py), or the coordinator's table."""
    busy = sw.live(node, t, sc.dt_s, pc.lost_after_s, d.i)
    cr = sw.current_round(lay, node, busy, max_rounds(pc))
    choice = None
    if pc.coordinator and d.i != 0:
        e = node.latest.get(0)
        fresh = e is not None and t - (e.step + 1) * sc.dt_s <= pc.lost_after_s
        if fresh and e.table is not None:
            nxt = e.table[d.i]
            if lay.is_lane(nxt):
                r = lay.round_of(nxt)
                if lay.lane_of(nxt) in sw.free_lanes(lay, node, busy, r) and (
                    cr is not None and r <= cr[0]
                ):
                    choice = nxt
            elif nxt != cp.NO_TASK and cr is None and nxt not in set(busy.values()):
                choice = nxt
    if choice is None and cr is not None:
        lane = sw.choose_lane(lay, d.i, cr[1])
        if lane is not None:
            choice = lay.lane_task(lane, cr[0])
    if choice is not None and lay.is_lane(choice):
        d.route = sw.lane_route(lay.lanes[lay.lane_of(choice)], d.pos)
        task[d.i] = choice
        return
    if choice is None:
        p, _ = smap.probabilities()
        rates = sw.block_rates(p, sc.grid, lay, d.pos, sc.speed_ms, spacing, sc.sensor.pd_per_pass)
        for tk in busy.values():
            if tk != cp.NO_TASK and not lay.is_lane(tk):
                rates[lay.block_of(tk)] = -1.0
        choice = lay.block_task(int(np.argmax(rates)))
    n_lo, n_hi, e_lo, e_hi = sim.block_bounds(sc.grid, lay.block_cells, lay.block_of(choice))
    d.route = sim.lanes(n_lo, n_hi, e_lo, e_hi, spacing, tuple(d.pos))
    task[d.i] = choice
