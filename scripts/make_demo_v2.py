"""Version 2 demo video (build2 Section 15): a person found and placed on the map, then a team
of drones sharing what they have searched. No words or numbers on screen (as D-618): the
story and the credits go in the text that accompanies it.

Usage: python scripts/make_demo_v2.py [--map cache/maps/syndrone_Town07_h50] [--seed N]

Scene 1, one drone (about 40 s): SynDrone Town07 at 50 m, a development town, from its search
map (scripts/make_search_map.py syndrone --town Town07 --height 50 --source v2_dfine; D-750).
Left, the flight's frames with each output track's box: orange for a track the simulation's
truth matches to a person, teal for one matched to a vehicle, grey for a false alarm or a near
miss (the map's truth labels). Right, the map: the ground the camera has covered, revealed as it
is searched (the flight's own frames laid on the ground), the drone's path and its current
footprint, and one marker per track at its latest ground position. Played at 1.5 times speed.

Scene 2, a team of four (about 37 s): one search of the Phase 14 simulation with the frozen
choice (configs/multi_drone.yaml: auction planner, 4-lane blocks, 20 m, sharing on, perfect
radio) and the camera measured on the SynDrone test towns (sensor_p10_v2_cal.json). The seed
is the development seed whose time to find is closest to the median of that configuration
over development seeds (results/multi_drone/budget_check_p10_v2_cal_seeds.json, `unlimited`),
so the search shown is a typical one, not the best. The frozen simulator is not changed: its
drones, maps and the person's position are recorded by wrapping them from here. Left, the team:
each drone's searched ground in its colour, its path and position, false alarms as grey dots,
the person as an orange marker once found. Right, the first drone's own map of where the person
may be (brighter is more likely): it darkens over ground other drones searched, because they
shared their observations. One simulated second per video frame.

Writes cache/video/demo_v2.mp4 (dataset imagery, not committed, D-323) and
results/video/demo_v2.json (sources, seed, duration, size, and whether it is within the limits:
20 MB and 90 s, build2 Section 15).
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


def typical_seed() -> tuple[int, float, float]:
    z = json.loads(
        (REPO_ROOT / "results/multi_drone/budget_check_p10_v2_cal_seeds.json").read_text()
    )["unlimited"]
    import multidrone_run as mr

    t, found = np.array(z["t_s"]), np.array(z["found"])
    # the budget check keeps the order of the development seeds (ProcessPoolExecutor.map)
    seed = np.array(z.get("seed", mr.DEV_SEEDS))
    if len(seed) != len(t):
        raise ValueError("per-seed outcomes and seeds differ in length")
    med = float(np.median(t))  # horizon counts for searches without a find, as in the results
    cand = np.nonzero(found)[0]
    best = cand[np.lexsort((seed[cand], np.abs(t[cand] - med)))[0]]
    return int(seed[best]), float(t[best]), med


def record(seed: int) -> dict[str, Any]:
    import multidrone_run as mr
    from aerialrange.search import belief, prior, sim
    from aerialrange.search import comms as cm

    st = mr.load_settings()
    if not st["frozen"]:
        raise SystemExit("configs/multi_drone.yaml is not frozen")
    sc = mr.scenario(
        st, REPO_ROOT / "results/multi_drone/sensor_p10_v2_cal.json", st["height"],
        st["n_drones"], "ring",
    )  # fmt: skip
    g = sc.grid
    rec: dict[str, Any] = {"pos": [], "yaw": [], "cover": [], "belief": [], "fa": []}
    drones: list = []
    maps: list = []
    owner = np.full(g.size, -1, np.int8)
    fa: list[tuple[float, float]] = []
    person: list = []
    orig_init, orig_step = sim.Drone.__init__, sim.Drone.step
    orig_from, orig_apply = belief.SearchMap.from_prior.__func__, belief.SearchMap.apply
    orig_sample = prior.sample_location

    def init(self: Any, *a: Any) -> None:
        orig_init(self, *a)
        drones.append(self)

    def snapshot() -> None:
        rec["pos"].append(np.array([d.pos for d in drones]))
        rec["yaw"].append(np.array([d.yaw for d in drones]))
        rec["cover"].append(owner.copy())
        rec["belief"].append(maps[0].probabilities()[0].astype(np.float32))
        rec["fa"].append(len(fa))

    def step(self: Any, *a: Any) -> None:
        if self.i == 0:
            snapshot()
        orig_step(self, *a)

    def from_prior(cls: Any, *a: Any) -> Any:
        m = orig_from(cls, *a)
        maps.append(m)
        return m

    def apply(self: Any, obs: Any) -> bool:
        if maps and self is maps[obs.drone]:  # the drone's own pass, applied first
            owner[np.asarray(obs.cells, np.int64)] = obs.drone
            fa.extend((d.n_m, d.e_m) for d in obs.detections)
        return orig_apply(self, obs)

    def sample(*a: Any) -> Any:
        p = orig_sample(*a)
        person.append(p)
        return p

    sim.Drone.__init__, sim.Drone.step = init, step
    belief.SearchMap.from_prior, belief.SearchMap.apply = classmethod(from_prior), apply
    prior.sample_location = sample
    try:
        pc = sim.PlannerConfig(kind=st["planner"], share=True, block_lanes=st["block_lanes"])
        res = sim.run(sc, pc, cm.Radio(), seed)
    finally:
        sim.Drone.__init__, sim.Drone.step = orig_init, orig_step
        belief.SearchMap.from_prior = classmethod(orig_from)
        belief.SearchMap.apply = orig_apply
        prior.sample_location = orig_sample
    snapshot()
    rec.update(grid=g, person=person[0], result=res, fa_points=fa, prior=sc.prior)
    return rec


def scene2(seed: int) -> tuple[Any, dict]:
    rec = record(seed)
    g = rec["grid"]
    side = 600
    boxes = [(30, 60), (W - 30 - side, 60)]
    s = side / g.ncols

    def to_px(n: float, e: float, b: tuple[int, int]) -> tuple[int, int]:
        return int(b[0] + (e - g.e0_m) / g.cell_m * s), int(
            b[1] + side - (n - g.n0_m) / g.cell_m * s
        )

    decades = 3.0  # each frame scaled to its own likeliest cells, three decades below them

    def heat(p: np.ndarray) -> np.ndarray:
        top = float(np.quantile(p, 0.999))  # not the peak: single false-alarm cells spike
        v = 1.0 + np.log10(np.maximum(p / top, 1e-12)) / decades
        u8 = (np.clip(v, 0, 1) * 255).astype(np.uint8).reshape(g.nrows, g.ncols)[::-1]
        return cv2.resize(
            cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO),
            (side, side),
            interpolation=cv2.INTER_NEAREST,
        )

    base = heat(rec["prior"])
    base = (0.12 * base + 0.88 * np.array(PANEL)).astype(np.uint8)
    pal = np.array(DRONES, np.uint8)
    n = len(rec["pos"])
    res = rec["result"]
    trails = [[to_px(*p, boxes[0]) for p in np.array(rec["pos"])[:, i]] for i in range(len(DRONES))]
    found_k = n - 1 if res.found else None
    fa_px = np.array([to_px(nm, em, boxes[0]) for nm, em in rec["fa_points"]], int).reshape(-1, 2)
    fa_px = np.clip(
        fa_px, [boxes[0][0], boxes[0][1]], [boxes[0][0] + side - 1, boxes[0][1] + side - 1]
    )

    def frames() -> Any:
        for k in range(n + int(HOLD_S * FPS)):
            kk = min(k, n - 1)
            img = np.full((H, W, 3), BG, np.uint8)
            # left: the team
            left = base.copy()
            cov = rec["cover"][kk].reshape(g.nrows, g.ncols)[::-1]
            col = cv2.resize(pal[np.maximum(cov, 0)], (side, side), interpolation=cv2.INTER_NEAREST)
            m = (
                cv2.resize(
                    (cov >= 0).astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
            left[m] = (0.6 * col[m] + 0.4 * left[m]).astype(np.uint8)
            x0, y0 = boxes[0]
            img[y0 : y0 + side, x0 : x0 + side] = left
            fx, fy = fa_px[: rec["fa"][kk]].T  # every detection so far, one dim pixel each
            img[fy, fx] = (0.5 * img[fy, fx] + 0.5 * np.array(GREY)).astype(np.uint8)
            for i, c in enumerate(DRONES):
                tr = np.array(trails[i][: kk + 1], np.int32)
                if len(tr) > 1:
                    cv2.polylines(img, [tr], False, c, 1, cv2.LINE_AA)
                p, yaw = rec["pos"][kk][i], rec["yaw"][kk][i]
                cx, cy = to_px(*p, boxes[0])
                fwd = np.array([np.sin(yaw), -np.cos(yaw)])  # east, north-up screen
                pts = np.array([fwd * 9, np.array([-fwd[1], fwd[0]]) * 5 - fwd * 5,
                                np.array([fwd[1], -fwd[0]]) * 5 - fwd * 5]) + [cx, cy]  # fmt: skip
                cv2.fillPoly(img, [pts.astype(np.int32)], c, cv2.LINE_AA)
            # right: the first drone's own map
            x1, y1 = boxes[1]
            img[y1 : y1 + side, x1 : x1 + side] = heat(rec["belief"][kk])
            cx, cy = to_px(*rec["pos"][kk][0], boxes[1])
            cv2.circle(img, (cx, cy), 5, DRONES[0], -1, cv2.LINE_AA)
            if found_k is not None and k >= found_k:
                age = k - found_k
                for b in boxes:
                    q = to_px(*rec["person"], b)
                    cv2.circle(img, q, 6, ORANGE, -1, cv2.LINE_AA)
                    cv2.circle(img, q, 8 + (age * 2) % 24, ORANGE, 1, cv2.LINE_AA)
            for b in boxes:
                cv2.rectangle(img, b, (b[0] + side, b[1] + side), (70, 70, 70), 1)
            yield img

    info = {
        "config": "auction, 4-lane blocks, 20 m, sharing on, perfect radio "
        "(configs/multi_drone.yaml, frozen)",
        "sensor": "results/multi_drone/sensor_p10_v2_cal.json",
        "seed": seed,
        "found": bool(res.found),
        "t_found_s": res.t_found_s,
        "false_alarms_before_find": res.false_alarms_before_find,
        "output_frames": n + int(HOLD_S * FPS),
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
    ap.add_argument("--seed", type=int, help="default: the typical development seed")
    ap.add_argument("--crf", type=int, default=26)
    args = ap.parse_args(argv)
    seed, t_seed, med = typical_seed()
    if args.seed is not None:
        seed = args.seed
    g1, i1 = scene1(REPO_ROOT / args.map)
    g2, i2 = scene2(seed)

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
        "scene2_team": {
            **i2,
            "seed_rule": "development seed whose time to find is closest to the median of the "
            "chosen configuration with sharing and no byte limit "
            "(budget_check_p10_v2_cal_seeds.json 'unlimited'); ties to the lower seed",
            "median_time_s": med,
            "seed_time_s_in_results": t_seed if args.seed is None else None,
        },
    }
    rv = REPO_ROOT / "results" / "video" / "demo_v2.json"
    rv.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("file", "seconds", "size_mb", "within_limits")}))
    return 0 if out["within_limits"] else 1


if __name__ == "__main__":
    sys.exit(main())
