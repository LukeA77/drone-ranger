"""Version 2 demo video: a person found and placed on the map, then two teams
of drones that each lose a drone. No words or numbers on screen (as D-618): the
story and the credits go in the text that accompanies it.

Usage: python scripts/make_demo_v2.py [--map cache/maps/syndrone_Town07_h50] [--seed N]

Scene 1, one drone (about 40 s): SynDrone Town07 at 50 m, a development town, from its search
map (scripts/make_search_map.py syndrone --town Town07 --height 50 --source v2_dfine; D-750).
Left, the flight's frames with each output track's box: orange for a track the simulation's
truth matches to a person, teal for one matched to a vehicle, grey for a false alarm or a near
miss (the map's truth labels). Right, the map: the ground the camera has covered, revealed as it
is searched (the flight's own frames laid on the ground), the drone's path and its current
footprint, and one marker per track at its latest ground position. Played at 1.5 times speed.

Scene 2, losing a drone (Phase 16, D-767): two searches of the Phase 16 simulation side by side,
on the same missing person and the same camera (measured on the SynDrone test towns, with the
recurring look-alikes of D-764), each with four drones of which the same one stops for good at
half a sweep's time (the test's `sweep_lose_any` and `lawnmower_lose_any`). Left, the sweep
planner on the frozen Phase 16 settings: the others take over the lost drone's lanes. Right, the
lawnmower: the lost drone's strip stays unsearched. Each panel: the ground each drone covered
first in its colour, paths and positions, the lost drone as a grey cross (a white ring as it
is lost), every detection as a dim grey dot, places reported to the crew that were not the
person as grey rings, and the person as an orange marker once confirmed (that panel then stops).
The seed is chosen on development seeds by a rule fixed before those searches ran (SEED_RULE),
so the comparison shown is a typical one where the loss matters, not the best. The frozen
engine is not changed: drones, alarms and the person are recorded by wrapping them from here.
Several simulated seconds per video frame, so that the whole video stays within 90 s.

Writes cache/video/demo_v2.mp4 (dataset imagery, not committed, D-323) and
results/video/demo_v2.json (sources, seed, duration, size, and whether it is within the limits:
20 MB and 90 s).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerialrange.utils.config import REPO_ROOT  # noqa: E402

W, H, FPS = 1280, 720, 25
MAX_MB, MAX_S = 20.0, 90.0
BG = (24, 20, 16)  # BGR, near black
PANEL = (34, 30, 26)
ORANGE, TEAL, GREY, WHITE = (40, 150, 255), (180, 190, 40), (150, 150, 150), (235, 235, 235)
DRONES = [(255, 170, 60), (90, 220, 120), (230, 110, 230), (60, 220, 230)]  # BGR
FADE_S, HOLD_S = 0.6, 2.0
SCENE1_SPEED = 1.5


# ---------------------------------------------------------------- scene 1


def load_map(folder: Path) -> dict[str, Any]:
    html = (folder / "map.html").read_text(encoding="utf-8").replace("<\\/", "</")
    data, _ = json.JSONDecoder().raw_decode(html, html.index('{"type":"FeatureCollection"'))
    return data


class MapView:
    """Lon/lat to panel pixels through the ground image's corners (an axis-aligned overlay)."""

    def __init__(self, gi: dict, ground: np.ndarray, box: tuple[int, int, int, int]) -> None:
        (self.lon0, self.lat0), (lon1, _), (_, lat1) = gi["tl"], gi["tr"], gi["bl"]
        self.dlon, self.dlat = lon1 - self.lon0, self.lat0 - lat1
        x0, y0, bw, bh = box
        gh, gw = ground.shape[:2]
        s = min(bw / gw, bh / gh)
        self.w, self.h = int(gw * s), int(gh * s)
        self.x0, self.y0 = x0 + (bw - self.w) // 2, y0 + (bh - self.h) // 2
        g = cv2.resize(ground, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.rgb = g[..., :3]
        self.alpha = (g[..., 3] > 0) if g.shape[2] == 4 else np.ones((self.h, self.w), bool)
        self.revealed = np.zeros((self.h, self.w), np.uint8)

    def px(self, lonlat: np.ndarray) -> np.ndarray:
        ll = np.asarray(lonlat, float).reshape(-1, 2)
        x = (ll[:, 0] - self.lon0) / self.dlon * self.w
        y = (self.lat0 - ll[:, 1]) / self.dlat * self.h
        return np.column_stack([x, y])

    def reveal(self, ring: list) -> None:
        cv2.fillPoly(self.revealed, [np.round(self.px(ring)).astype(np.int32)], 1)

    def draw_ground(self, img: np.ndarray) -> None:
        roi = img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        roi[:] = PANEL
        m = (self.revealed > 0) & self.alpha
        roi[m] = self.rgb[m]
        faint = ~m & self.alpha  # ground the flight will reach, shown very dim
        roi[faint] = (0.12 * self.rgb[faint] + 0.88 * np.array(PANEL)).astype(np.uint8)

    def to_img(self, p: np.ndarray) -> np.ndarray:
        return np.round(p + [self.x0, self.y0]).astype(np.int32)


def scene1(folder: Path) -> tuple[list[np.ndarray] | Any, dict]:
    data = load_map(folder)
    dr = data["drone_ranger"]
    match = dr["truth_match"]
    vframes = [int(f) for f in dr["video"]["frames"]]
    ground = cv2.imread(str(folder / "ground.webp"), cv2.IMREAD_UNCHANGED)
    view = MapView(dr["ground_image"], ground, (816, 40, 440, 640))
    vid_box = (24, 144, 768, 432)  # 0.4 of the 1920 x 1080 frames
    scale = vid_box[2] / float(dr["video"]["width"])
    feats = data["features"]
    foot = {
        int(f["properties"]["frame"]): f["geometry"]["coordinates"][0]
        for f in feats
        if f["properties"]["kind"] == "footprint"
    }
    dets: dict[int, list[tuple[str, list]]] = {}
    for f in feats:
        p = f["properties"]
        if p["kind"] == "detection":
            dets.setdefault(int(p["frame"]), []).append(
                (p["track_id"], f["geometry"]["coordinates"])
            )
    track_cls = {
        f["properties"]["track_id"]: f["properties"]["class"]
        for f in feats
        if f["properties"]["kind"] == "track"
    }
    drone = view.to_img(view.px(np.array(dr["drone_track"])))
    timeline = [int(f) for f, _ in dr["timeline"]]
    cap = cv2.VideoCapture(str(folder / "video.mp4"))
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(vframes):
        raise ValueError("the video and the map list different numbers of frames")
    cur: list = [-1, None]  # index and image of the last frame read (read in order, not kept)

    def src_frame(k: int) -> np.ndarray:
        while cur[0] < k:
            ok, fr = cap.read()
            if not ok:
                raise ValueError(f"video ends before frame {k}")
            cur[0], cur[1] = cur[0] + 1, fr
        return cv2.resize(cur[1], vid_box[2:], interpolation=cv2.INTER_AREA)

    def colour(tid: str) -> tuple[tuple[int, int, int], int]:
        m = match.get(tid, "false alarm")
        if m != "real":
            return GREY, 1
        return (ORANGE, 2) if track_cls.get(tid) == "person" else (TEAL, 2)

    n_out = int(len(vframes) / SCENE1_SPEED)
    latest: dict[str, np.ndarray] = {}
    first_seen: dict[str, int] = {}
    done = 0  # pipeline frames applied to the map so far

    def frames() -> Any:
        nonlocal done
        for j in range(n_out):
            k = min(int(j * SCENE1_SPEED), len(vframes) - 1)
            f = vframes[k]
            while done < f:  # every pipeline frame up to this one, in order
                done += 1
                if done in foot:
                    view.reveal(foot[done])
                for tid, ll in dets.get(done, []):
                    latest[tid] = view.to_img(view.px(ll))[0]
                    first_seen.setdefault(tid, j)
            img = np.full((H, W, 3), BG, np.uint8)
            x, y, bw, bh = vid_box
            img[y : y + bh, x : x + bw] = src_frame(k)
            for b in dr["boxes"].get(str(f), []):
                tid, l_, t_, w_, h_ = str(b[0]), *b[1:5]
                c, th = colour(tid)
                p0 = (int(x + l_ * scale), int(y + t_ * scale))
                p1 = (int(x + (l_ + w_) * scale), int(y + (t_ + h_) * scale))
                cv2.rectangle(img, p0, p1, c, th, cv2.LINE_AA)
            view.draw_ground(img)
            i = timeline.index(f) if f in timeline else 0
            cv2.polylines(img, [drone[: i + 1]], False, WHITE, 1, cv2.LINE_AA)
            if f in foot:
                ring = view.to_img(view.px(foot[f]))
                cv2.polylines(img, [ring], True, WHITE, 1, cv2.LINE_AA)
            for tid, p in latest.items():
                c, _ = colour(tid)
                person = c == ORANGE
                if c == GREY:
                    cv2.circle(img, tuple(p), 3, GREY, 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, tuple(p), 5 if person else 3, c, -1, cv2.LINE_AA)
                if person:  # a person newly placed on the map: one expanding ring
                    age = j - first_seen[tid]
                    if age < FPS:
                        r = 6 + int(18 * age / FPS)
                        cv2.circle(img, tuple(p), r, ORANGE, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(drone[i]), 4, WHITE, -1, cv2.LINE_AA)
            yield img

    info = {
        "source": folder.relative_to(REPO_ROOT).as_posix(),
        "flight": dr.get("title"),
        "split": "development",
        "speed": SCENE1_SPEED,
        "output_frames": n_out,
        "tracks_by_truth_match": {
            k: sum(v == k for v in match.values()) for k in sorted(set(match.values()))
        },
    }
    return frames(), info


# ---------------------------------------------------------------- scene 2

SEED_FILE = REPO_ROOT / "results" / "video" / "demo_v2_seed_choice.json"
SEED_RULE = (
    "development seeds 0 to 199, four drones, primary camera, perfect radio, one drone (drawn "
    "per seed) lost at half a sweep's time, frozen Phase 16 settings; candidates: seeds where "
    "the person is in the lost drone's lawnmower strip, on ground no lawnmower drone had "
    "covered when it was lost (the situation the lost-drone result is about; decided from the "
    "person's position and the lost drone only, both drawn before any detection, and from the "
    "lawnmower's fixed routes); chosen: the candidate whose lawnmower-minus-sweep time to a "
    "confirmed find (the horizon when none) is closest to the median of that difference over "
    "the candidates; ties to the lower seed (D-767)."
)
# The first rule, fixed before the searches ran, kept for the record: it chose seed 133, where the
# person was outside the lost drone's ground, so both teams searched alike and the takeover never
# appears (D-767).
FIRST_RULE = (
    "candidates: seeds where both the sweep and the lawnmower confirm the find after the loss; "
    "chosen: the candidate whose lawnmower-minus-sweep time is closest to the median of that "
    "difference over the candidates; ties to the lower seed."
)


def configs16() -> dict[str, dict]:
    """The test's two lost-drone settings (multidrone16_run.py: sweep_lose_any and
    lawnmower_lose_any), on the frozen Phase 16 choices."""
    import multidrone16_run as r16
    import multidrone_run as mr

    st, s16 = mr.load_settings(), r16.load16()
    if not s16["frozen"]:
        raise SystemExit("configs/search16.yaml is not frozen")
    sw_pc = r16.sweep_pc(
        s16["spacing_factor"], s16["post"], s16["block_lanes"], s16["lost_after_s"]
    )
    b = {"camera": "primary", "n": st["n_drones"], "k": s16["confirm_k"]}
    b["m"] = s16["min_track_frames"]
    return {
        "sweep": {**b, "pc": sw_pc, "lose": "any"},
        "lawnmower": {**b, "pc": {"kind": "lawnmower"}, "lose": "any"},
    }


def choose_seed(workers: int) -> dict[str, Any]:
    """Run both settings on the development seeds (once; kept in SEED_FILE), apply SEED_RULE."""
    import multidrone16_run as r16
    import multidrone_run as mr

    cfgs = configs16()
    have = json.loads(SEED_FILE.read_text(encoding="utf-8")) if SEED_FILE.exists() else None
    if have is None or have.get("settings") != cfgs:
        from concurrent.futures import ProcessPoolExecutor

        st = mr.load_settings()
        jobs = [(st, c, s) for c in cfgs.values() for s in mr.DEV_SEEDS]
        print(f"seed choice: {len(jobs)} development searches, {workers} workers", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(r16.job, jobs, chunksize=4))
        n = len(mr.DEV_SEEDS)
        have = {
            "rule": SEED_RULE,
            "settings": cfgs,
            "rows": {lab: rows[i * n : (i + 1) * n] for i, lab in enumerate(cfgs)},
        }
    rows = have["rows"]
    loss_t = rows["sweep"][0]["horizon"] / mr.HORIZON_COVERS / 2
    in_lost = lost_drone_ground(cfgs["lawnmower"], loss_t, [r["seed"] for r in rows["sweep"]])

    def t(r: dict) -> float:
        return r["t"] if r["found"] else r["horizon"]

    cand, first = [], []
    for a, b in zip(rows["sweep"], rows["lawnmower"], strict=True):
        if a["seed"] != b["seed"]:
            raise ValueError("seed order differs between settings")
        if in_lost[a["seed"]]:
            cand.append((a["seed"], t(b) - t(a), a["t"], b["t"]))
        if a["found"] and b["found"] and min(a["t"], b["t"]) > loss_t:
            first.append((a["seed"], b["t"] - a["t"]))
    if not cand:
        raise SystemExit("no candidate seed")
    med = float(np.median([c[1] for c in cand]))
    seed, diff, t_sw, t_lm = min(cand, key=lambda c: (abs(c[1] - med), c[0]))
    med1 = float(np.median([c[1] for c in first]))
    have["first_rule"] = {
        "rule": FIRST_RULE,
        "seed": min(first, key=lambda c: (abs(c[1] - med1), c[0]))[0],
        "n_candidates": len(first),
        "median_lawnmower_minus_sweep_s": med1,
    }
    have["rule"] = SEED_RULE
    have["choice"] = {
        "seed": seed,
        "loss_time_s": loss_t,
        "n_candidates": len(cand),
        "n_seeds": len(rows["sweep"]),
        "median_lawnmower_minus_sweep_s": med,
        "lawnmower_minus_sweep_s": diff,
        "sweep_t_s": t_sw,
        "lawnmower_t_s": t_lm,
    }
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEED_FILE.write_text(json.dumps(have, indent=1) + "\n", encoding="utf-8")
    return have["choice"]


def lost_drone_ground(c: dict, loss_t: float, seeds: list[int]) -> dict[int, bool]:
    """Seed -> whether the person is in the lost drone's lawnmower strip, on ground no lawnmower
    drone had covered by the loss. The lawnmower's routes do not depend on the seed, so they are
    flown here with the simulation's own drone, planner and camera footprint up to the loss; the
    person and the lost drone are the engine's own draws for each seed."""
    import multidrone16_run as r16
    import multidrone_run as mr
    from aerialrange.search import prior, sim

    st = mr.load_settings()
    sc = mr.scenario(st, r16.camera_files(c["camera"], c["m"])[0], st["height"], c["n"], "ring")
    g, s = sc.grid, sc.sensor
    spacing = sim.lane_spacing_m(s)
    pc = sim.PlannerConfig("lawnmower", True, 4)
    seen = np.zeros(g.size, bool)
    for i in range(c["n"]):
        d = sim.Drone(i, *sc.launch_ne)
        for k in range(int(np.ceil(loss_t / sc.dt_s))):
            t = (k + 1) * sc.dt_s
            if t > loss_t:  # the engine stops a lost drone once t passes the loss time
                break
            if not d.route:
                sim.plan(d, pc, sc, None, {}, [], spacing, 1, t)
            d.step(sc.speed_ms, sc.dt_s)
            fp = s.footprint(float(d.pos[0]), float(d.pos[1]), d.yaw)
            if len(fp):
                seen[np.asarray(g.cells_in_polygon(fp), np.int64)] = True
    width = g.ncols * g.cell_m / c["n"]
    out = {}
    for seed in seeds:
        n_m, e_m = prior.sample_location(sc.prior, g, np.random.default_rng([seed, 0]))
        who = int(np.random.default_rng([seed, 4]).integers(c["n"]))
        strip = min(int((e_m - g.e0_m) // width), c["n"] - 1)
        out[seed] = strip == who and not seen[int(g.cell_of(n_m, e_m))]
    return out


def record(c: dict, seed: int) -> dict[str, Any]:
    """One search of the frozen Phase 16 engine, set up as multidrone16_run.job sets it up, with
    the drones' positions, the ground each drone covered first, every detection and every alarm
    recorded by wrapping them from here (the engine is not changed)."""
    import multidrone16_run as r16
    import multidrone_run as mr
    from aerialrange.search import comms as cm
    from aerialrange.search import confirm, prior, sim
    from aerialrange.search import engine as en

    st = mr.load_settings()
    sensor, share, fire, radius = r16.camera_files(c["camera"], c["m"])
    sc = mr.scenario(st, sensor, st["height"], c["n"], "ring")
    who = int(np.random.default_rng([seed, 4]).integers(c["n"]))  # as job() draws "any"
    lose = (who, sc.horizon_s / mr.HORIZON_COVERS / 2)
    world = en.World(
        persistent_share=share, persistent_fire=fire, confirm_k=c["k"],
        confirm_radius_m=radius, stop="confirmed", lose=lose,
    )  # fmt: skip
    g, s = sc.grid, sc.sensor
    drones: list = []
    pos: list[np.ndarray] = []
    yaw: list[np.ndarray] = []
    first = np.full(g.size, -1, np.int32)  # step at which a cell was first covered
    owner = np.full(g.size, -1, np.int8)
    dets: list[tuple[float, float, int]] = []
    alarms: list[tuple[float, float, int, bool]] = []
    person: list = []
    last = [10**9]
    orig_init, orig_step = sim.Drone.__init__, sim.Drone.step
    orig_add, orig_sample = confirm.Alarms.add, prior.sample_location

    def init(self: Any, *a: Any) -> None:
        orig_init(self, *a)
        drones.append(self)

    def step(self: Any, *a: Any) -> None:
        if self.i <= last[0]:  # a new simulation step begins
            pos.append(np.array([d.pos for d in drones]))
            yaw.append(np.array([d.yaw for d in drones]))
        last[0] = self.i
        orig_step(self, *a)
        k = len(pos) - 1
        fp = s.footprint(float(self.pos[0]), float(self.pos[1]), self.yaw)
        if len(fp):
            cells = np.asarray(g.cells_in_polygon(fp), np.int64)
            new = cells[first[cells] < 0]
            first[new], owner[new] = k, self.i

    def add(self: Any, n: float, e: float, t: float, p: Any) -> Any:
        k = int(round(t / sc.dt_s)) - 1
        dets.append((n, e, k))
        r = orig_add(self, n, e, t, p)
        if r is not None:
            alarms.append((n, e, k, bool(r)))
        return r

    def sample(*a: Any) -> Any:
        p = orig_sample(*a)
        person.append(p)
        return p

    sim.Drone.__init__, sim.Drone.step = init, step
    confirm.Alarms.add, prior.sample_location = add, sample
    try:
        res = en.run(sc, en.Plan(**c["pc"]), cm.Radio(), seed, world)
    finally:
        sim.Drone.__init__, sim.Drone.step = orig_init, orig_step
        confirm.Alarms.add, prior.sample_location = orig_add, orig_sample
    pos.append(np.array([d.pos for d in drones]))
    yaw.append(np.array([d.yaw for d in drones]))
    return {
        "grid": g, "prior": sc.prior, "dt": sc.dt_s, "pos": np.array(pos), "yaw": np.array(yaw),
        "first": first, "owner": owner, "dets": np.array(dets, float).reshape(-1, 3),
        "alarms": alarms, "person": person[0], "result": res,
        "lose": (who, int(round(lose[1] / sc.dt_s)) - 1),
    }  # fmt: skip


def scene2(choice: dict, max_frames: int) -> tuple[Any, dict]:
    """Left the sweep, right the lawnmower: the same missing person, the same drone lost."""
    cfgs = configs16()
    seed = choice["seed"]
    recs = [record(cfgs["sweep"], seed), record(cfgs["lawnmower"], seed)]
    for r, lab in zip(recs, ("sweep", "lawnmower"), strict=True):
        if r["result"].t_confirmed_s != choice[f"{lab}_t_s"]:
            raise SystemExit(f"{lab}: the recorded search differs from the seed-choice run")
    g = recs[0]["grid"]
    side = 600
    boxes = [(30, 60), (W - 30 - side, 60)]
    s = side / g.ncols
    ends = [len(r["pos"]) - 1 for r in recs]
    hold = int(HOLD_S * FPS)
    per_frame = max(1, int(np.ceil(max(ends) / max(max_frames - hold - 1, 1))))
    n_out = int(np.ceil(max(ends) / per_frame)) + 1 + hold

    def to_px(n: float, e: float, b: tuple[int, int]) -> tuple[int, int]:
        return int(b[0] + (e - g.e0_m) / g.cell_m * s), int(
            b[1] + side - (n - g.n0_m) / g.cell_m * s
        )

    top = float(np.quantile(recs[0]["prior"], 0.999))
    v = 1.0 + np.log10(np.maximum(recs[0]["prior"] / top, 1e-12)) / 3.0
    u8 = (np.clip(v, 0, 1) * 255).astype(np.uint8).reshape(g.nrows, g.ncols)[::-1]
    base = cv2.resize(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), (side, side),
                      interpolation=cv2.INTER_NEAREST)  # fmt: skip
    base = (0.12 * base + 0.88 * np.array(PANEL)).astype(np.uint8)
    pal = np.array(DRONES, np.uint8)
    draw = []
    for r, b in zip(recs, boxes, strict=True):
        trails = [[to_px(*p, b) for p in r["pos"][:, i]] for i in range(r["pos"].shape[1])]
        dpx = np.array([to_px(n_, e_, b) for n_, e_, _ in r["dets"]], int).reshape(-1, 2)
        dpx = np.clip(dpx, [b[0], b[1]], [b[0] + side - 1, b[1] + side - 1])
        crew = [(to_px(n_, e_, b), k) for n_, e_, k, ok in r["alarms"] if not ok]
        draw.append((trails, dpx, crew))

    def panel(img: np.ndarray, j: int, k_out: int) -> None:
        r, b = recs[j], boxes[j]
        trails, dpx, crew = draw[j]
        k = min(k_out, ends[j])
        x0, y0 = b
        cov = ((r["first"] >= 0) & (r["first"] <= k)).reshape(g.nrows, g.ncols)[::-1]
        own = r["owner"].reshape(g.nrows, g.ncols)[::-1]
        m = cv2.resize(cov.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST) > 0
        col = cv2.resize(pal[np.maximum(own, 0)], (side, side), interpolation=cv2.INTER_NEAREST)
        tile = base.copy()
        tile[m] = (0.6 * col[m] + 0.4 * tile[m]).astype(np.uint8)
        img[y0 : y0 + side, x0 : x0 + side] = tile
        nd = int(np.searchsorted(r["dets"][:, 2], k, side="right"))
        fx, fy = dpx[:nd].T  # every detection so far, one dim pixel each
        img[fy, fx] = (0.5 * img[fy, fx] + 0.5 * np.array(GREY)).astype(np.uint8)
        # places reported to the crew that were not the person: faint rings, so that the
        # ground each drone covered stays readable under thousands of them
        win = img[y0 : y0 + side, x0 : x0 + side]
        ov = win.copy()
        for (qx, qy), ka in crew:
            if ka <= k:
                cv2.circle(ov, (qx - x0, qy - y0), 3, GREY, 1, cv2.LINE_AA)
        win[:] = cv2.addWeighted(ov, 0.35, win, 0.65, 0)
        lost, k_lost = r["lose"]
        for i, c in enumerate(DRONES[: r["pos"].shape[1]]):
            gone = i == lost and k > k_lost
            kk = min(k, k_lost) if i == lost else k
            tr = np.array(trails[i][: kk + 1], np.int32)
            if len(tr) > 1:
                cv2.polylines(img, [tr], False, c, 1, cv2.LINE_AA)
            cx, cy = to_px(*r["pos"][kk][i], b)
            if gone:
                cv2.line(img, (cx - 6, cy - 6), (cx + 6, cy + 6), GREY, 2, cv2.LINE_AA)
                cv2.line(img, (cx - 6, cy + 6), (cx + 6, cy - 6), GREY, 2, cv2.LINE_AA)
                age = (k - k_lost) / per_frame  # output frames since the loss
                if age < FPS:  # a widening ring for one second after the loss
                    cv2.circle(img, (cx, cy), 10 + int(age * 1.5), WHITE, 1, cv2.LINE_AA)
                continue
            yw = r["yaw"][kk][i]
            fwd = np.array([np.sin(yw), -np.cos(yw)])  # east, north-up screen
            pts = np.array([fwd * 9, np.array([-fwd[1], fwd[0]]) * 5 - fwd * 5,
                            np.array([fwd[1], -fwd[0]]) * 5 - fwd * 5]) + [cx, cy]  # fmt: skip
            cv2.fillPoly(img, [pts.astype(np.int32)], c, cv2.LINE_AA)
        if r["result"].confirmed and k_out >= ends[j]:
            age = (k_out - ends[j]) // per_frame
            q = to_px(*r["person"], b)
            cv2.circle(img, q, 6, ORANGE, -1, cv2.LINE_AA)
            cv2.circle(img, q, 8 + (age * 2) % 24, ORANGE, 1, cv2.LINE_AA)
        cv2.rectangle(img, b, (b[0] + side, b[1] + side), (70, 70, 70), 1)

    def frames() -> Any:
        for f in range(n_out):
            img = np.full((H, W, 3), BG, np.uint8)
            for j in range(2):
                panel(img, j, f * per_frame)
            yield img

    def outcome(r: dict) -> dict:
        x = r["result"]
        return {
            "confirmed": x.confirmed,
            "t_confirmed_s": x.t_confirmed_s,
            "crew_alarms_before_find": x.crew_alarms_before_confirmed,
        }

    info = {
        "config": "left: sweep first then look again (frozen Phase 16 settings, "
        "configs/search16.yaml); right: lawnmower; four drones, primary camera, perfect radio, "
        "the same drone lost at half a sweep's time in both",
        "sensor": "results/multi_drone/sensor_p10_v2_cal.json, with the recurring look-alikes "
        "of clutter_p10_v2_cal.json",
        "seed": seed,
        "lost_drone": recs[0]["lose"][0],
        "lost_at_s": (recs[0]["lose"][1] + 1) * recs[0]["dt"],
        "sweep": outcome(recs[0]),
        "lawnmower": outcome(recs[1]),
        "simulated_seconds_per_frame": per_frame * recs[0]["dt"],
        "output_frames": n_out,
    }
    return frames(), info


# ---------------------------------------------------------------- output


def fade(gen: Any, n_total: int) -> Any:
    nf = int(FADE_S * FPS)
    for j, img in enumerate(gen):
        a = min(1.0, (j + 1) / nf, (n_total - j) / nf)
        yield img if a >= 1 else (img * a + np.array(BG) * (1 - a)).astype(np.uint8)


def encode(frames: Any, dest: Path, crf: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(dest)]  # fmt: skip
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    assert p.stdin is not None
    for img in frames:
        p.stdin.write(img.tobytes())
        n += 1
    p.stdin.close()
    if p.wait():
        raise SystemExit("ffmpeg failed")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", default="cache/maps/syndrone_Town07_h50")
    ap.add_argument("--workers", type=int, default=5, help="for the seed choice (once)")
    ap.add_argument("--crf", type=int, default=26)
    args = ap.parse_args(argv)
    choice = choose_seed(args.workers)
    g1, i1 = scene1(REPO_ROOT / args.map)
    budget = int((MAX_S - 1.0) * FPS) - i1["output_frames"]
    g2, i2 = scene2(choice, budget)

    def all_frames() -> Any:
        yield from fade(g1, i1["output_frames"])
        yield from fade(g2, i2["output_frames"])

    dest = REPO_ROOT / "cache" / "video" / "demo_v2.mp4"
    n = encode(all_frames(), dest, args.crf)
    mb = dest.stat().st_size / 1e6
    out = {
        "file": dest.relative_to(REPO_ROOT).as_posix(),
        "seconds": round(n / FPS, 2),
        "size_mb": round(mb, 2),
        "size": [W, H],
        "encoding": f"H.264 crf {args.crf}",
        "limits": {"max_mb": MAX_MB, "max_s": MAX_S},
        "within_limits": mb <= MAX_MB and n / FPS <= MAX_S,
        "on_screen_text": "none (D-618)",
        "scene1_one_drone": i1,
        "scene2_lost_drone": {
            **i2,
            "seed_rule": SEED_RULE,
            "seed_choice": {k: v for k, v in choice.items() if k != "seed"},
            "seed_choice_file": SEED_FILE.relative_to(REPO_ROOT).as_posix(),
        },
    }
    rv = REPO_ROOT / "results" / "video" / "demo_v2.json"
    rv.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("file", "seconds", "size_mb", "within_limits")}))
    return 0 if out["within_limits"] else 1


if __name__ == "__main__":
    sys.exit(main())
