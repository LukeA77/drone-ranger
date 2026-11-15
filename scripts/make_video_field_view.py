"""The 3D field-view video: one flight and the team search, drawn as 3D scenes.

Usage: python scripts/make_video_field_view.py [--stills 1,5,12 ...] [--only 1|2]

Chapter 1: the drone flies over the map it is building, seen from an orbiting camera: its camera
cone sweeps the ground, the ground image appears where it has searched, and a pin drops for each
person and vehicle. The drone's own camera view sits in a corner. One person is held while the
view circles: a line joins their box in the picture to their pin on the ground.
Chapter 2: the Phase 16 lost-drone search (the seed and recordings of make_video_team.py) on a
tilted map: numbered drones on stems, one goes silent and sinks, the others cover its ground.

Writes cache/video/field_view.mp4 (not committed) and results/video/field_view.json (what it
shows and where each number comes from). The README's loops are cut from it by
scripts/make_readme_media.py. Every number comes from the search map, the recorded simulation or
a results file. Stems and pins are drawn taller than life so they can be seen.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_kit as kit  # noqa: E402
from video_kit import DRONES, FPS, GREY, ORANGE, RED, ROOT, H, W, ease, fade, font, lerp, rgb  # noqa: E402

OUT = ROOT / "cache" / "video"
CRF = 29  # keeps the film under GitHub's 10 MB upload limit
BG_TOP, BG_BOT = (26, 20, 14), (40, 32, 24)  # BGR
INK, INK2 = (245, 245, 245), (170, 176, 182)  # RGB
TRACK, T_FREEZE, ALT = "14", 2.76, 50.0
INSET = (24, 24, 416, 234)
F_CAP, F_VAL, F_LAB = font(28, "semibold"), font(27, "bold"), font(11, "bold")
F_TAG, F_TAGB, F_NOTE, F_NUM = font(15), font(17, "bold"), font(13), font(14, "bold")

CAP1 = {
    "see": "A drone's camera sees pixels: boxes in a picture.",
    "place": "Drone Ranger works out where each one is on the ground.\n"
    "This person: {rng:.0f} m from the drone, placed to within {size:.1f} m.",
    "pins": "Every person and vehicle gets a pin, a distance and an error bar.",
    "map": "The map is built from the camera's own frames. No internet, no map service.",
    "all": "Two minutes of flight: {people} people and {vehicles} vehicles pinned, "
    "{km2:.2f} km² searched.",
    "false": "Grey rings: {false} of the {people} were look-alikes, not people. That is the open\n"
    "problem, and a thermal camera is the next step.",
}
CAP2 = {
    "team": "Now a team. Four drones search {km2:.1f} km² for one missing person.",
    "lost": "Drone {lost} goes silent: a flat battery, a crash or a lost radio.\n"
    "The others cannot tell which.",
    "count": "After {after:.0f} seconds of silence they count it as lost,\n"
    "and add the ground it left to their own.",
    "found": "They fly that ground and see someone. A second sighting confirms it.",
    "pair": "Without the team plan each drone keeps to its own lanes,\n"
    "and the lost drone's ground is never searched.",
}


def finish(img: np.ndarray, text: str, alpha: float, note: str, draw: list) -> np.ndarray:
    """Bottom shade, then all text: tags and HUD (draw: callables taking an ImageDraw)."""
    kit.bottom_shade(img, 190, 0.92)
    pil = kit.to_pil(img)
    d = ImageDraw.Draw(pil)
    for fn in draw:
        fn(d)
    if text and alpha > 0:
        y = 652 if "\n" not in text else 634
        kit.shadow_text(d, (W // 2, y), text, F_CAP, INK, anchor="ma", alpha=alpha, align="center")
    if note:
        d.text((W - 28, 84), note, font=F_NOTE, fill=(130, 138, 146), anchor="ra")
    return kit.to_bgr(pil)


def hud(items: list[tuple[str, str]], warn: int | None = None):  # noqa: ANN201
    def fn(d: ImageDraw.ImageDraw) -> None:
        x = W - 28
        for n in range(len(items) - 1, -1, -1):
            lab, val = items[n]
            w = max(d.textlength(val, font=F_VAL), d.textlength(lab, font=F_LAB)) + 34
            col = (255, 120, 120) if n == warn else INK
            kit.shadow_text(d, (x, 22), val, F_VAL, col, anchor="ra")
            d.text((x, 58), lab, font=F_LAB, fill=INK2, anchor="ra")
            x -= int(w)

    return fn


def tag(x: int, y: int, lines: list[str], col: tuple, side: int = 1):  # noqa: ANN201
    def fn(d: ImageDraw.ImageDraw) -> None:
        w = int(max(d.textlength(s, font=F_TAGB if i == 0 else F_TAG) for i, s in enumerate(lines)))
        h = 12 + 21 * len(lines)
        left = x + 16 if side > 0 else x - 16 - w - 20
        left = min(max(left, 8), W - w - 28)
        top = min(max(y - h - 8, 84), H - h - 120)
        d.rounded_rectangle((left, top, left + w + 20, top + h), 6, fill=(16, 20, 26), outline=col)
        for i, s in enumerate(lines):
            d.text(
                (left + 10, top + 6 + 21 * i),
                s,
                font=F_TAGB if i == 0 else F_TAG,
                fill=INK if i == 0 else INK2,
            )

    return fn


def keyed(keys: list[tuple], u: float) -> list[float]:
    """Piecewise eased interpolation of (u, values...) keyframes."""
    if u <= keys[0][0]:
        return list(keys[0][1:])
    for a, b in zip(keys, keys[1:], strict=False):
        if u <= b[0]:
            x = ease((u - a[0]) / (b[0] - a[0]))
            return [lerp(p, q, x) for p, q in zip(a[1:], b[1:], strict=True)]
    return list(keys[-1][1:])


def pin(img: np.ndarray, cam: kit.Cam, g: np.ndarray, col: tuple, drop: float, big: bool) -> tuple:
    """A pin standing on ground point g; drop in [0, 1] is its fall. Returns the head's pixel."""
    pxm = cam.px_per_m([g[0], g[1], 0])
    h = max(4.0, (22 if big else 13) / pxm)
    top = [g[0], g[1], h + (1 - ease(drop)) * 28 / pxm]
    a, b = cam.pt([g[0], g[1], 0]), cam.pt(top)
    if drop >= 1:
        cv2.line(img, a, b, col, 2 if big else 1, cv2.LINE_AA)
        cv2.circle(img, a, 2, col, -1, cv2.LINE_AA)
    r = 6 if big else 4
    cv2.circle(img, b, r + 1, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, b, r, col, -1, cv2.LINE_AA)
    return b


def quad(img: np.ndarray, cam: kit.Cam, p: np.ndarray, yaw: float) -> None:
    """A small quadcopter at p (east, north, up), white on a dark edge."""
    pxm = cam.px_per_m(p)
    arm = max(2.4, 11 / pxm)
    r = max(3, int(0.42 * arm * pxm))
    ends = [
        p + arm * np.array([np.sin(yaw + a), np.cos(yaw + a), 0])
        for a in np.radians([45, 135, 225, 315])
    ]
    c = cam.pt(p)
    for col, th in (((20, 20, 20), 5), ((245, 245, 245), 2)):
        for e in ends:
            cv2.line(img, c, cam.pt(e), col, th, cv2.LINE_AA)
    for e in ends:
        q = cam.pt(e)
        cv2.circle(img, q, r + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, q, r, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.circle(img, c, max(3, r // 2 + 1), ORANGE, -1, cv2.LINE_AA)


# ------------------------------------------------------------------ chapter 1

SEGS1 = [  # name, flight seconds from, to, video seconds, captions
    ("fly", 0.0, T_FREEZE, 3.2, ["see"]),
    ("freeze", T_FREEZE, T_FREEZE, 5.4, ["place"]),
    ("fly", T_FREEZE, 60.0, 11.0, ["pins", "map"]),
    ("fly", 60.0, 119.9, 5.4, ["all"]),
    ("hold", 119.9, 119.9, 5.2, ["false"]),
]
# video second, distance, azimuth, pitch, how far the view follows the drone, look-at height
CAM1 = [
    (0.0, 150, -28, 31, 1.0, 22),
    (3.2, 132, -16, 30, 1.0, 22),
    (8.6, 132, 24, 30, 1.0, 22),
    (13.5, 300, 34, 40, 1.0, 12),
    (19.6, 420, 44, 47, 0.6, 0),
    (25.0, 780, 12, 66, 0.0, 0),
    (30.2, 820, 2, 70, 0.0, 0),
]


def chapter1(fl: kit.Flight, gt: kit.GroundTex) -> Iterator[np.ndarray]:
    bg = kit.vgrad(BG_TOP, BG_BOT)
    centre = np.array([fl.lon0 + fl.dlon / 2, fl.lat0 - fl.dlat / 2]) * kit.M_PER_DEG
    obj = fl.objs[TRACK]
    person = np.array(obj["ll"]) * kit.M_PER_DEG
    final = fl.counts(119.96)
    fmt = {"rng": obj["range_m"], "size": obj["ellipse_major_m"], **final}
    first_j: dict[str, int] = {}
    note = "Simulated town (SynDrone). The drone's position is known exactly here."
    u0, j = 0.0, 0
    ix, iy, iw, ih = INSET
    for name, t0, t1, secs, keys in SEGS1:
        n = int(secs * FPS)
        per = n // len(keys)
        for f in range(n):
            j += 1
            u = u0 + f / FPS
            t = lerp(t0, t1, f / n)
            i = int(min(round(t * 25), len(fl.frames) - 1))
            frame, k = fl.frames[i], fl.k_at(t)
            gt.advance(frame)
            dist, az, pitch, follow, zt = keyed(CAM1, u)
            dg = fl.drone_m[i]
            target = np.array([*lerp(centre, dg, follow), zt])
            cam = kit.orbit(target, dist, az, pitch)
            img = bg.copy()
            tex, t2w = gt.level(dist / cam.f > 0.34)
            cam.ground(img, tex, t2w, fog=(dist * 2.4, dist * 4.2))
            # path flown, on the ground
            if i > 1:
                path = np.column_stack(
                    [fl.drone_m[: i + 1 : 3], np.zeros(len(fl.drone_m[: i + 1 : 3]))]
                )
                uv, z = cam.project(path)
                ok = z > 1
                cv2.polylines(
                    img, [np.round(uv[ok]).astype(np.int32)], False, (235, 235, 235), 1, cv2.LINE_AA
                )
            # the camera's cone and its footprint
            dp = np.array([dg[0], dg[1], ALT])
            ring = fl.foot.get(frame)
            if ring is not None and name != "hold":
                rw = np.column_stack([ring * kit.M_PER_DEG, np.zeros(len(ring))])
                cam.fill(img, rw, (255, 255, 255), 0.10)
                cam.line(img, rw, (255, 255, 255), 1, closed=True)
                hull = cv2.approxPolyDP(rw[:, :2].astype(np.float32).reshape(-1, 1, 2), 6.0, True)
                for c in hull.reshape(-1, 2):
                    cam.line(img, [dp, [c[0], c[1], 0]], (200, 200, 200), 1)
            # pins: grey rings for false alarms, pins for people and vehicles the truth confirms
            pulse = name == "hold"
            heads = {}
            for tid, o in fl.objs.items():
                if o["first_t_s"] > t + 1e-9:
                    continue
                first_j.setdefault(tid, j)
                g = np.array(o["ll"]) * kit.M_PER_DEG
                col = fl.colour(tid)
                if col == GREY:
                    r = 3 + (int(2.5 * (1 + np.sin(j / FPS * 7.5))) if pulse else 0)
                    c = (235, 235, 235) if pulse else GREY
                    cv2.circle(img, cam.pt([g[0], g[1], 0]), r, c, 2 if pulse else 1, cv2.LINE_AA)
                    continue
                age = (j - first_j[tid]) / (0.35 * FPS)
                heads[tid] = pin(img, cam, g, col, min(age, 1.0), col == ORANGE)
            # the drone, its shadow and stem
            if name != "hold":
                cv2.circle(img, cam.pt([dg[0], dg[1], 0]), 4, (20, 20, 20), -1, cv2.LINE_AA)
                cam.line(img, [[dg[0], dg[1], 0], dp], (120, 120, 120), 1)
                nxt = fl.drone_m[min(i + 12, len(fl.drone_m) - 1)] - fl.drone_m[max(i - 12, 0)]
                quad(img, cam, dp, float(np.arctan2(nxt[0], nxt[1])))
            draw = [
                hud(
                    [
                        ("PEOPLE FOUND", f"{fl.counts(t)['people']}"),
                        ("VEHICLES FOUND", f"{fl.counts(t)['vehicles']}"),
                        ("GROUND SEARCHED", f"{fl.counts(t)['km2']:.2f} km²"),
                        ("FLIGHT TIME", f"{int(t // 60)}:{int(t % 60):02d}"),
                    ]
                )
            ]
            # the drone's camera view, in the corner
            a_in = 1.0 if u < 24.2 else max(0.0, 1 - (u - 24.2) / 0.6)
            if a_in > 0:
                small = cv2.resize(fl.frame(k), (iw, ih), interpolation=cv2.INTER_AREA)
                frozen = name == "freeze"
                if frozen:
                    small = (small * 0.5).astype(np.uint8)
                s = iw * fl.box_scale
                mine = None
                for b in fl.dr["boxes"].get(str(fl.vframes[k]), []):
                    tid = str(b[0])
                    p0 = (int(b[1] * s), int(b[2] * s))
                    p1 = (int((b[1] + b[3]) * s), int((b[2] + b[4]) * s))
                    if frozen and tid == TRACK:
                        mine = (p0, p1)
                        continue
                    cv2.rectangle(small, p0, p1, fl.colour(tid), 1, cv2.LINE_AA)
                if mine:
                    src = cv2.resize(fl.frame(k), (iw, ih), interpolation=cv2.INTER_AREA)
                    (x0, y0), (x1, y1) = mine
                    small[y0 - 4 : y1 + 4, x0 - 4 : x1 + 4] = src[y0 - 4 : y1 + 4, x0 - 4 : x1 + 4]
                    cv2.rectangle(small, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), ORANGE, 2, cv2.LINE_AA)
                roi = img[iy : iy + ih, ix : ix + iw]
                roi[:] = (small * a_in + roi * (1 - a_in)).astype(np.uint8)
                cv2.rectangle(img, (ix - 1, iy - 1), (ix + iw, iy + ih), (110, 110, 110), 1)
                draw.append(
                    lambda d: d.text(
                        (ix + 8, iy + ih + 6), "WHAT THE DRONE'S CAMERA SEES", font=F_LAB, fill=INK2
                    )
                )
                if mine and TRACK in heads:
                    grow = ease(f / (0.7 * FPS))
                    # line of sight, drone to person, then picture to pin
                    gp = np.array([person[0], person[1], 0.0])
                    cam.line(img, [dp, dp + (gp - dp) * grow], ORANGE, 2)
                    if grow >= 1:
                        g2 = ease((f - 0.7 * FPS) / (0.6 * FPS))
                        bx = np.array([ix + (mine[0][0] + mine[1][0]) / 2, iy + mine[1][1] + 4])
                        hd = np.array(heads[TRACK], float)
                        e = bx + (hd - bx) * g2
                        cv2.line(
                            img, tuple(bx.astype(int)), tuple(e.astype(int)), ORANGE, 1, cv2.LINE_AA
                        )
                        cam.ring(img, gp, 2.2, ORANGE, 2)
                        if g2 >= 1:
                            hx, hy = heads[TRACK]
                            draw.append(
                                tag(
                                    hx,
                                    hy,
                                    [
                                        "Person",
                                        f"{obj['range_m']:.0f} m from the drone",
                                        f"95% area: {obj['ellipse_major_m']:.1f} × "
                                        f"{obj['ellipse_minor_m']:.1f} m",
                                    ],
                                    rgb(ORANGE),
                                )
                            )
            m = min(f // per, len(keys) - 1)
            text = CAP1[keys[m]].format(**fmt)
            alpha = 1.0 if j <= 8 else fade(f - m * per, per if m < len(keys) - 1 else n - m * per)
            yield finish(img, text, alpha, note, draw)
        u0 += n / FPS


# ------------------------------------------------------------------ chapter 2


def team_searches() -> tuple:
    """The two recorded searches of make_video_team.py (cached: recording takes half a minute)."""
    cache = ROOT / "cache" / "video" / "field_view_team.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    import make_video_team as vt
    from make_demo_v2 import choose_seed, configs16, record

    choice = choose_seed(5)
    cfgs = configs16()
    after = float(cfgs["sweep"]["pc"]["lost_after_s"])
    out = (
        vt.Search(record(cfgs["sweep"], choice["seed"]), after),
        vt.Search(record(cfgs["lawnmower"], choice["seed"]), after),
        choice,
        after,
    )
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


TEX = 660
UNSEARCHED = (36, 32, 28)
STEM = 120.0  # metres: drones are drawn on stems far taller than their 20 m, to be seen


def team_texture(s, k: int, j: int) -> np.ndarray:  # noqa: ANN001
    k = min(k, s.end)
    cov = (s.first >= 0) & (s.first <= k)
    tile = np.full((*cov.shape, 3), UNSEARCHED, np.uint8)
    tile[cov] = (0.66 * s.colour[cov] + 0.34 * np.array(UNSEARCHED)).astype(np.uint8)
    tex = cv2.resize(tile, (TEX, TEX), interpolation=cv2.INTER_NEAREST)
    step = TEX / 11  # a line every 200 m
    for a in range(1, 11):
        p = int(round(a * step))
        dark = ~cv2.resize(
            cov.astype(np.uint8), (TEX, TEX), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        tex[p, dark[p]] = (54, 50, 46)
        tex[dark[:, p], p] = (54, 50, 46)
    if k >= s.k_lost:
        rest = cv2.morphologyEx(
            (s.left & ~cov).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        m = cv2.resize(rest, (TEX, TEX), interpolation=cv2.INTER_NEAREST) > 0
        yy, xx = np.nonzero(m)
        if len(yy):
            a = 0.6 + 0.25 * np.sin(j / FPS * 2 * np.pi * 0.9)
            st = ((xx + yy) // 7) % 2 == 0
            tex[yy[st], xx[st]] = (tex[yy[st], xx[st]] * (1 - a) + 240 * a).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tex, cs, -1, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.rectangle(tex, (0, 0), (TEX - 1, TEX - 1), (120, 116, 112), 2)
    return tex


def team_scene(
    img: np.ndarray,
    cam: kit.Cam,
    s,  # noqa: ANN001
    k: int,
    j: int,
    sink: float,
    never: bool = False,
    small: bool = False,
) -> list:
    """One tilted search map with its drones; returns text callables."""
    g = s.g
    span = g.ncols * g.cell_m
    t2w = np.array([[span / TEX, 0, g.e0_m], [0, -span / TEX, g.n0_m + span], [0, 0, 1.0]])
    k = min(k, s.end)
    cam.ground(img, team_texture(s, k, j), t2w, keep_black=False)
    draw: list = []
    rad = 9 if small else 12

    def w3(n: float, e: float, z: float = 0.0) -> list[float]:
        return [e, n, z]

    for i, c in enumerate(DRONES[: s.n]):
        if i == s.lost and k > s.k_lost:
            continue
        n_, e_ = s.pos[k][i]
        k0 = max(k - 45, 0)
        trail = np.array([w3(*p) for p in s.pos[k0 : k + 1, i]])
        if len(trail) > 1:
            cam.line(img, trail, tuple(int(0.5 * v + 127) for v in c), 2)
        a, b = cam.pt(w3(n_, e_)), cam.pt(w3(n_, e_, STEM))
        cv2.circle(img, a, 3, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.line(img, a, b, c, 2, cv2.LINE_AA)
        cv2.circle(img, b, rad + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, b, rad, c, -1, cv2.LINE_AA)
        draw.append(
            lambda d, b=b, i=i: d.text(b, str(i + 1), font=F_NUM, fill=(20, 20, 20), anchor="mm")
        )
    if k > s.k_lost:
        n_, e_ = s.pos[s.k_lost][s.lost]
        age = (k - s.k_lost) * s.dt
        if age < 100 and not small:
            for ph in (0.0, 0.5):
                r = 60 + 260 * ((j / FPS * 0.9 + ph) % 1.0)
                cam.ring(img, w3(n_, e_), r, RED, 2)
        a, b = cam.pt(w3(n_, e_)), cam.pt(w3(n_, e_, STEM * (1 - ease(sink))))
        cv2.line(img, a, b, GREY, 2, cv2.LINE_AA)
        cv2.circle(img, b, rad + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, b, rad, GREY, -1, cv2.LINE_AA)
        for dx in (-1, 1):
            cv2.line(
                img,
                (b[0] - 5, b[1] - 5 * dx),
                (b[0] + 5, b[1] + 5 * dx),
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
        if not small:
            draw.append(tag(b[0], b[1], [f"Drone {s.lost + 1} silent"], (255, 120, 120)))
    # the missing person
    pn, pe = s.person
    seen = s.k_seen is not None and k >= s.k_seen
    done = s.res.confirmed and k >= s.end
    a, b = cam.pt(w3(pn, pe)), cam.pt(w3(pn, pe, 80.0))
    cv2.line(img, a, b, ORANGE, 2, cv2.LINE_AA)
    cv2.circle(img, b, 8, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, b, 6, ORANGE, -1, cv2.LINE_AA)
    if done or seen:
        for ph in (0.0, 0.5):
            r = 40 + (200 if done else 90) * ((j / FPS * 1.1 + ph) % 1.0)
            cam.ring(img, w3(pn, pe), r, ORANGE, 2)
    label = (
        "Found and confirmed" if done else "Seen once" if seen else "Never seen" if never else None
    )
    if label is None and k < 400 and not small:
        label = "The missing person"
    if label:
        col = (255, 120, 120) if never else rgb(ORANGE)
        draw.append(tag(b[0], b[1], [label], col, side=-1 if small and not never else 1))
    return draw


SEGS2 = [  # name, from event, to event, video seconds, caption
    ("search", "start", "lost", 6.0, "team"),
    ("silence", "lost", "presumed", 4.6, "lost"),
    ("own lanes", "presumed", "takeover", 4.6, "count"),
    ("takeover", "takeover", "confirmed", 5.2, "found"),
]
HOLD2, PAIR_PLAY, PAIR_HOLD, CARD = 2.6, 4.0, 3.4, 5.0
CAM2 = [  # video second, distance, azimuth, pitch, target east, target north
    (0.0, 3500, -30, 42, 0, -80),
    (6.0, 3400, -14, 43, 0, -80),
    (10.6, 3250, 0, 44, 120, -80),
    (15.2, 3250, 12, 45, 150, -60),
    (20.4, 2850, 22, 47, 250, -40),
    (23.0, 2800, 27, 47, 250, -40),
]


def chapter2(team, fixed, after: float) -> Iterator[np.ndarray]:  # noqa: ANN001
    bg = kit.vgrad(BG_TOP, BG_BOT)
    marks = team.marks()
    fmt = {"km2": team.km2, "lost": team.lost + 1, "after": after}
    note = "Simulation. The camera's hit and false-alarm rates were measured on rendered towns."
    u0, j = 0.0, 0
    for name, a, b, secs, key in SEGS2:
        n = int(secs * FPS)
        hold = int(HOLD2 * FPS) if name == "takeover" else 0
        for f in range(n + hold):
            j += 1
            u = u0 + f / FPS
            k = marks[b] if f >= n else int(round(lerp(marks[a], marks[b], f / n)))
            dist, az, pitch, te, tn = keyed(CAM2, u)
            cam = kit.orbit([te, tn, 0], dist, az, pitch)
            img = bg.copy()
            sink = (k - team.k_lost) * team.dt / 14.0
            draw = team_scene(img, cam, team, k, j, sink)
            flying = team.n - (1 if k > team.k_lost else 0)
            t = team.t(k)
            draw.append(
                hud(
                    [
                        ("SEARCH TIME", f"{int(t // 60)}:{int(t % 60):02d}"),
                        ("GROUND SEARCHED", f"{team.searched(k):.0%}"),
                        ("DRONES FLYING", f"{flying} of {team.n}"),
                        ("PLACES REPORTED TO THE CREW", f"{team.reported(k):,}"),
                    ],
                    2 if k > team.k_lost else None,
                )
            )
            yield finish(img, CAP2[key].format(**fmt), fade(f, n + hold), note, draw)
        u0 += (n + hold) / FPS
    # side by side
    n, hold = int(PAIR_PLAY * FPS), int(PAIR_HOLD * FPS)
    for f in range(n + hold):
        j += 1
        k = fixed.end if f >= n else int(round(fixed.end * f / n))
        img = bg.copy()
        draw = []
        for s, kk, cx, nv in ((team, team.end, 330, False), (fixed, k, 950, f >= n)):
            cam = kit.orbit([0, -150, 0], 5100, 0, 50, cx=cx, cy=400)
            draw += team_scene(img, cam, s, kk, j, 1.0, never=nv, small=True)
        tf = fixed.t(k)
        heads = [
            (
                330,
                "With the team plan",
                f"Found after {team.t(team.end) / 60:.0f} minutes",
                rgb(ORANGE),
            ),
            (
                950,
                "Without it",
                f"Not found after {tf / 60:.0f} minutes: search called off"
                if f >= n
                else f"Searching: {int(tf // 60)}:{int(tf % 60):02d}",
                (255, 120, 120) if f >= n else INK2,
            ),
        ]

        def titles(d: ImageDraw.ImageDraw, heads: list = heads) -> None:
            for cx, h1, h2, col in heads:
                kit.shadow_text(d, (cx, 40), h1, font(30, "bold"), INK, anchor="ma")
                kit.shadow_text(d, (cx, 84), h2, font(20, "semibold"), col, anchor="ma")

        draw.append(titles)
        draw.append(
            lambda d: d.text((W // 2, 150), note, font=F_NOTE, fill=(130, 138, 146), anchor="ma")
        )
        yield finish(img, CAP2["pair"], fade(f, n + hold), "", draw)


def card(cost: dict, n_test: int) -> Iterator[np.ndarray]:
    bg = kit.vgrad(BG_TOP, BG_BOT)
    lines = [
        ("Drone Ranger", font(64, "bold"), INK, 190),
        (
            "Turns a drone's camera into a map of the people and vehicles below:\n"
            "where each one is, and how sure it is.",
            font(25),
            (220, 224, 228),
            290,
        ),
        (
            f"Losing a drone added {cost['sweep']:,.0f} s to a search with the team plan, and "
            f"{cost['lawnmower']:,.0f} s without it,\nover {n_test} simulated test searches.",
            font(21),
            rgb(ORANGE),
            400,
        ),
        (
            "Research prototype: tested on public datasets and in simulation, not yet flown.",
            font(19),
            INK2,
            500,
        ),
        ("github.com/LukeA77/drone-ranger", font(19, "semibold"), INK2, 560),
    ]
    for f in range(int(CARD * FPS)):
        a = min(1.0, (f + 1) / (0.5 * FPS))
        pil = kit.to_pil(bg)
        d = ImageDraw.Draw(pil)
        for s, fnt, col, y in lines:
            c = tuple(int(v * a + 20 * (1 - a)) for v in col)
            d.multiline_text(
                (W // 2, y), s, font=fnt, fill=c, anchor="ma", align="center", spacing=8
            )
        yield kit.to_bgr(pil)


def dip(gen: Iterator[np.ndarray], n: int, secs: float = 0.4) -> Iterator[np.ndarray]:
    """Fade a chapter in from and out to the background colour."""
    k = int(secs * FPS)
    for f, img in enumerate(gen):
        a = min(1.0, (f + 1) / k if f > 0 else 1.0, (n - f) / k)
        yield img if a >= 1 else (img * a + np.array(BG_TOP) * (1 - a)).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default="", help="seconds, comma separated: save frames only")
    ap.add_argument("--only", default="", help="1 or 2: render one chapter (for stills)")
    args = ap.parse_args()
    stills = [float(x) for x in args.stills.split(",") if x]
    n1 = sum(int(s[3] * FPS) for s in SEGS1)
    n2 = (
        sum(int(s[3] * FPS) for s in SEGS2)
        + int(HOLD2 * FPS)
        + int(PAIR_PLAY * FPS)
        + int(PAIR_HOLD * FPS)
    )

    def frames() -> Iterator[np.ndarray]:
        if args.only != "2":
            fl = kit.Flight(ROOT / "cache/maps/syndrone_Town07_h50")
            yield from dip(chapter1(fl, kit.GroundTex(fl)), n1)
        if args.only != "1":
            team, fixed, _, after = team_searches()
            yield from dip(chapter2(team, fixed, after), n2)
            paired = json.loads((ROOT / "results/multi_drone/phase16_paired_test.json").read_text())
            cost = {
                k: paired["comparisons"][f"{k}_lose_any_minus_no_loss"]["time"]["mean"]
                for k in ("sweep", "lawnmower")
            }
            yield from card(cost, 300)

    dest = OUT / "field_view.mp4"
    kit.run(frames(), dest, stills, crf=CRF)
    if stills:
        return
    fl = kit.Flight(ROOT / "cache/maps/syndrone_Town07_h50")
    team, fixed, choice, after = team_searches()
    obj = fl.objs[TRACK]
    rec = {
        "file": dest.relative_to(ROOT).as_posix(),
        "seconds": round((n1 + n2 + int(CARD * FPS)) / FPS, 2),
        "size_mb": round(dest.stat().st_size / 1e6, 2),
        "size": [W, H],
        "encoding": f"H.264 crf {CRF}",
        "flight": {
            "source": "cache/maps/syndrone_Town07_h50 (SynDrone Town07, 50 m), "
            "as results/video/demo_one_drone.json",
            "person_shown": TRACK,
            "range_m": obj["range_m"],
            "ellipse_major_m": obj["ellipse_major_m"],
            "counts_at_end": {k: v for k, v in fl.counts(119.96).items() if k != "t_s"},
        },
        "team": {
            "source": "the two recorded searches of scripts/make_video_team.py (same rule)",
            "seed": choice["seed"],
            "lost_drone_shown_as": team.lost + 1,
            "presumed_lost_after_s": after,
            "team_plan_confirmed_s": team.t(team.end),
            "fixed_lanes_searched_until_s": fixed.t(fixed.end),
        },
        "end_card_source": "results/multi_drone/phase16_paired_test.json",
        "drawn_larger_than_life": "pins, drones and drone stems",
        "captions": {**CAP1, **{"team_" + k: v for k, v in CAP2.items()}},
    }
    (ROOT / "results/video/field_view.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
