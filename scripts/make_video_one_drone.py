"""Video 1, one drone: from the camera to the map, with captions and a live counter.

Usage: python scripts/make_video_one_drone.py [--map cache/maps/syndrone_Town07_h50] [--crf 26]

Made to be understood with the sound off by someone who knows nothing about the project, so,
unlike the earlier demo (make_demo_v2.py, no words on screen), it carries captions, panel titles,
a legend and a counter. Source: the search map of SynDrone Town07 at 50 m, a development town
(scripts/make_search_map.py syndrone --town Town07 --height 50 --source v2_dfine).

Layout: along the top, the counter of the map page (people found, vehicles found, ground
searched, flight time) and, under it, the people checked against the simulation's truth (the
map's own labels: real; near a real person, a real person inside the find's 95 % ellipse, so a
crew sent there would find them; false alarm; not scored, a track the evaluation ignores).
Left, the camera's frames with every track's box; right, the map built from those frames.
Colours: orange a person and teal a vehicle that the truth confirms (real or near), grey a
false alarm or a track not scored.

Sequence (25 frames per second):
1. the first FLIGHT_A_S seconds of the flight at OPEN_SPEED;
2. a freeze on one real person: their box, a magnified view, and a line to their place on the
   map with its distance from the drone and its 95 % ellipse (the map's values);
3. the flight on to FLIGHT_C_END_S at real speed, the counter highlighted;
4. a skip to the end of the 2-minute flight, then the whole path and the ground searched;
5. the false alarms highlighted, with their cause and what would fix them;
6. an end card pointing to video 2.

Counts at a moment are the objects first seen by then (the map's first_t_s), so the last frame
matches the map page. Ground searched at a moment is the map's searched_km2 times the share of
the final searched ground revealed by then (the footprints drawn on the panel).

Writes cache/video/demo_one_drone.mp4 (dataset imagery, not committed) and
results/video/demo_one_drone.json (source, segments, captions, final counts, size, limits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerialrange.utils.config import REPO_ROOT  # noqa: E402
from make_demo_v2 import (  # noqa: E402
    BG,
    FPS,
    GREY,
    ORANGE,
    TEAL,
    WHITE,
    H,
    MapView,
    W,
    encode,
    load_map,
)

MAX_MB, MAX_S = 10.0, 60.0  # GitHub plays an uploaded video up to 10 MB
SPEED = 1.0  # real time
OPEN_SPEED = 0.5  # the opening seconds, slowed so the first boxes can be followed
FLIGHT_A_S = 2.5  # opening stretch of flight, ending on the frozen person
FREEZE_TRACK = "14"  # a real person seen at FLIGHT_A_S: every box "person", every range reliable
FREEZE_S = 5.0
FLIGHT_C_END_S = 22.0
SKIP_S, WHOLE_S, FALSE_S, CARD_S = 1.4, 6.0, 10.0, 3.5
CAPTION_FADE_S = 0.3

VID = (24, 140, 768, 432)  # x, y, w, h: 0.4 of the 1920 x 1080 frames
MAP = (816, 140, 440, 432)
TILE = (40, 36, 32)
INK2 = (175, 170, 165)  # secondary text, BGR

CAPTIONS = {
    "a1": "A drone flies over a town, 50 m up. Its camera spots people and vehicles.",
    "a2": "Orange: a person.   Teal: a vehicle.   Grey: something that only looks like a person.",
    "b": "Each find is placed on the map. This person: {range} m from the drone, placed\n"
    "to within {size} m (simulation, with the drone's exact position).",
    "c1": "The map is built from the drone's own camera: no internet, no map service.",
    "c2": "The counter keeps score as it flies: people, vehicles and ground searched.",
    "d": "Skipping ahead to the end of the 2-minute flight",
    "e": "The whole flight: the path it flew, and all the ground it checked.",
    "f1": "Grey: things the camera mistook for a person. It learned from city streets\n"
    "filmed from lower down, where people look bigger.",
    "f2": "The crew confirms or dismisses each find on the map. The fix being built:\n"
    "a thermal camera, where people show up warm against cool ground.",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    p = win / ("segoeuib.ttf" if bold else "segoeui.ttf")
    if not p.exists():
        import matplotlib

        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        p = p / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(p), size)


def rgb(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (c[2], c[1], c[0])


F_LABEL, F_VALUE, F_SMALL = font(13, True), font(28, True), font(15)
F_TITLE, F_CAPTION, F_LEGEND = font(16, True), font(25), font(14)


class Flight:
    """The map's data, advanced frame by frame; draws the camera and map panels."""

    def __init__(self, folder: Path) -> None:
        data = load_map(folder)
        dr = self.dr = data["drone_ranger"]
        self.match = dr["truth_match"]
        self.vframes = [int(f) for f in dr["video"]["frames"]]
        self.t_of = {int(f): float(t) for f, t in dr["timeline"]}
        ground = cv2.imread(str(folder / "ground.webp"), cv2.IMREAD_UNCHANGED)
        self.view = MapView(dr["ground_image"], ground, MAP)
        self.scale = VID[2] / float(dr["video"]["width"])
        feats = data["features"]
        self.foot = {
            int(f["properties"]["frame"]): f["geometry"]["coordinates"][0]
            for f in feats
            if f["properties"]["kind"] == "footprint"
        }
        self.dets: dict[int, list[tuple[str, list]]] = {}
        for f in feats:
            p = f["properties"]
            if p["kind"] == "detection":
                self.dets.setdefault(int(p["frame"]), []).append(
                    (str(p["track_id"]), f["geometry"]["coordinates"])
                )
        self.objs = {
            str(f["properties"]["track_id"]): f["properties"]
            for f in feats
            if f["properties"]["kind"] == "object"
        }
        self.drone = self.view.to_img(self.view.px(np.array(dr["drone_track"])))
        self.timeline = [int(f) for f, _ in dr["timeline"]]
        # the final searched ground on the panel, for the live area
        final = np.zeros_like(self.view.revealed)
        for ring in self.foot.values():
            cv2.fillPoly(final, [np.round(self.view.px(ring)).astype(np.int32)], 1)
        self.final_px = int(final.sum())
        self.cap = cv2.VideoCapture(str(folder / "video.mp4"))
        if int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(self.vframes):
            raise ValueError("the video and the map list different numbers of frames")
        self.cur: list = [-1, None]
        self.done = 0
        self.latest: dict[str, np.ndarray] = {}
        self.first_out: dict[str, int] = {}
        self.out_j = 0

    # -- state
    def label(self, tid: str) -> str:
        return self.match.get(tid, "not scored")

    def colour(self, tid: str) -> tuple[tuple[int, int, int], int]:
        if self.label(tid) in ("real", "near"):
            cls = self.objs.get(tid, {}).get("class")
            return (ORANGE, 2) if cls == "person" else (TEAL, 2)
        return GREY, 1

    def advance(self, f: int, j: int) -> None:
        while self.done < f:
            self.done += 1
            if self.done in self.foot:
                self.view.reveal(self.foot[self.done])
            for tid, ll in self.dets.get(self.done, []):
                self.latest[tid] = self.view.to_img(self.view.px(ll))[0]
                self.first_out.setdefault(tid, j)

    def frame(self, k: int) -> np.ndarray:
        while self.cur[0] < k:
            ok, fr = self.cap.read()
            if not ok:
                raise ValueError(f"video ends before frame {k}")
            self.cur[0], self.cur[1] = self.cur[0] + 1, fr
        return self.cur[1]

    def t(self, k: int) -> float:
        return self.t_of[self.vframes[k]]

    def k_at(self, t_s: float) -> int:
        return int(np.argmin([abs(self.t(k) - t_s) for k in range(len(self.vframes))]))

    def counts(self, t_s: float) -> dict[str, Any]:
        seen = [o for o in self.objs.values() if o["first_t_s"] <= t_s + 1e-9]
        people = [str(o["track_id"]) for o in seen if o["class"] == "person"]
        km2 = self.dr["searched_km2"] * float(self.view.revealed.sum()) / self.final_px
        lab = [self.label(t) for t in people]
        return {
            "people": len(people),
            "vehicles": sum(o["class"] != "person" for o in seen),
            "km2": km2,
            "t_s": t_s,
            **{k: lab.count(k) for k in ("real", "near", "false alarm", "not scored")},
        }

    # -- drawing
    def draw(self, k: int, j: int, *, pulse_grey: bool = False) -> np.ndarray:
        f = self.vframes[k]
        img = np.full((H, W, 3), BG, np.uint8)
        x, y, bw, bh = VID
        img[y : y + bh, x : x + bw] = cv2.resize(self.frame(k), (bw, bh), cv2.INTER_AREA)
        for b in self.dr["boxes"].get(str(f), []):
            tid, l_, t_, w_, h_ = str(b[0]), *b[1:5]
            c, th = self.colour(tid)
            p0 = (int(x + l_ * self.scale), int(y + t_ * self.scale))
            p1 = (int(x + (l_ + w_) * self.scale), int(y + (t_ + h_) * self.scale))
            cv2.rectangle(img, p0, p1, c, th, cv2.LINE_AA)
        self.view.draw_ground(img)
        if pulse_grey:  # the ground steps back so the false alarms stand out
            v = self.view
            roi = img[v.y0 : v.y0 + v.h, v.x0 : v.x0 + v.w]
            roi[:] = (roi * 0.4).astype(np.uint8)
        i = self.timeline.index(f) if f in self.timeline else 0
        cv2.polylines(img, [self.drone[: i + 1]], False, WHITE, 1, cv2.LINE_AA)
        if f in self.foot:
            ring = self.view.to_img(self.view.px(self.foot[f]))
            cv2.polylines(img, [ring], True, WHITE, 1, cv2.LINE_AA)
        r_grey = 3
        if pulse_grey:
            r_grey = 4 + int(2.5 * (1 + np.sin(j / FPS * 2 * np.pi * 1.2)))
        for tid, p in self.latest.items():
            c, _ = self.colour(tid)
            if c == GREY:
                col = (225, 225, 225) if pulse_grey else GREY
                cv2.circle(img, tuple(p), r_grey, col, 2 if pulse_grey else 1, cv2.LINE_AA)
        for tid, p in self.latest.items():  # people and vehicles drawn over the grey
            c, _ = self.colour(tid)
            if c == GREY:
                continue
            person = c == ORANGE
            cv2.circle(img, tuple(p), 5 if person else 3, c, -1, cv2.LINE_AA)
            age = j - self.first_out.get(tid, -10 * FPS)
            if person and 0 <= age < FPS:  # newly placed: one expanding ring
                cv2.circle(img, tuple(p), 6 + int(18 * age / FPS), ORANGE, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(self.drone[i]), 4, WHITE, -1, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------- text and overlays


def fmt_time(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def chrome(img: np.ndarray, c: dict, *, glow: float = 0.0) -> Image.Image:
    """Counter, truth line, panel titles and legends; returns a PIL image to draw captions on."""
    tiles = [
        ("PEOPLE FOUND", f"{c['people']}"),
        ("VEHICLES FOUND", f"{c['vehicles']}"),
        ("GROUND SEARCHED", f"{c['km2']:.2f} km²"),
        ("FLIGHT TIME", fmt_time(c["t_s"])),
    ]
    pil = Image.fromarray(img[..., ::-1])
    d = ImageDraw.Draw(pil)
    gap, x0, tw = 12, 24, (W - 48 - 3 * 12) // 4
    for n, (lab, val) in enumerate(tiles):
        x = x0 + n * (tw + gap)
        d.rounded_rectangle((x, 12, x + tw, 74), 8, fill=rgb(TILE))
        if glow > 0:
            pairs = zip(rgb(ORANGE), rgb(TILE), strict=True)
            col = tuple(int(v * glow + b * (1 - glow)) for v, b in pairs)
            d.rounded_rectangle((x, 12, x + tw, 74), 8, outline=col, width=3)
        d.text((x + 14, 20), lab, font=F_LABEL, fill=rgb(INK2))
        d.text((x + 14, 34), val, font=F_VALUE, fill=(245, 245, 245))
    truth = (
        f"People checked against the simulation's truth:   {c['real']} real   ·   "
        f"{c['near']} near a real person   ·   {c['false alarm']} false alarms"
    )
    if c["not scored"]:
        truth += f"   ·   {c['not scored']} not scored"
    d.text((W // 2, 92), truth, font=F_SMALL, fill=rgb(INK2), anchor="mt")
    d.text((VID[0], 116), "What the drone's camera sees", font=F_TITLE, fill=(230, 230, 230))
    d.text((MAP[0], 116), "The map it builds", font=F_TITLE, fill=(230, 230, 230))
    ly = VID[1] + VID[3] + 12
    x = VID[0]
    for col, name in ((ORANGE, "person"), (TEAL, "vehicle"), (GREY, "look-alike (false alarm)")):
        d.rectangle((x, ly + 3, x + 12, ly + 15), outline=rgb(col), width=2)
        d.text((x + 18, ly), name, font=F_LEGEND, fill=rgb(INK2))
        x += 26 + int(d.textlength(name, font=F_LEGEND)) + 14
    x = MAP[0]
    d.line((x, ly + 9, x + 16, ly + 9), fill=(235, 235, 235), width=2)
    d.text((x + 22, ly), "path flown", font=F_LEGEND, fill=rgb(INK2))
    x += 22 + int(d.textlength("path flown", font=F_LEGEND)) + 16
    d.rectangle((x, ly + 3, x + 12, ly + 15), fill=(150, 140, 110))
    d.text((x + 18, ly), "ground searched", font=F_LEGEND, fill=rgb(INK2))
    return pil


def caption(pil: Image.Image, text: str, alpha: float) -> np.ndarray:
    if text and alpha > 0:
        d = ImageDraw.Draw(pil)
        v = int(245 * alpha + BG[0] * (1 - alpha))
        y = 650 if "\n" in text else 664  # one line centred in the band, two lines from its top
        d.multiline_text(
            (W // 2, y - (16 if "\n" in text else 0)), text, font=F_CAPTION, fill=(v, v, v),
            anchor="ma", align="center", spacing=6,
        )  # fmt: skip
    return np.array(pil)[..., ::-1].copy()


def cap_alpha(j: int, n: int) -> float:
    nf = CAPTION_FADE_S * FPS
    return float(min(1.0, (j + 1) / nf, (n - j) / nf))


# ---------------------------------------------------------------- segments


def play(
    fl: Flight,
    k0: int,
    k1: int,
    cues: list[tuple[float, str]],
    glow_from: float | None,
    speed: float = SPEED,
) -> Iterator[np.ndarray]:
    """Frames k0..k1 at `speed`; cues are (start second within the segment, caption key)."""
    step = fl.dr["video"]["fps"] / FPS * speed
    n = int((k1 - k0) / step)
    for j in range(n):
        k = min(k0 + int(j * step), k1)
        fl.out_j += 1  # one clock across segments, for the newly-placed rings
        fl.advance(fl.vframes[k], fl.out_j)
        img = fl.draw(k, fl.out_j)
        s = j / FPS
        key, start, end = None, 0.0, n / FPS
        for m, (t0, kk) in enumerate(cues):
            if s >= t0:
                key, start = kk, t0
                end = cues[m + 1][0] if m + 1 < len(cues) else n / FPS
        glow = 0.0
        if glow_from is not None and s >= glow_from:
            glow = 0.6 + 0.4 * np.sin((s - glow_from) * 2 * np.pi * 0.8)
        pil = chrome(img, fl.counts(fl.t(k)), glow=glow)
        a = cap_alpha(int((s - start) * FPS), int((end - start) * FPS))
        yield caption(pil, CAPTIONS[key] if key else "", a)


def freeze(fl: Flight, k: int, tid: str) -> Iterator[np.ndarray]:
    """One frame held: the person's box, a magnified view, a line to the map and a label."""
    f = fl.vframes[k]
    box = next(b for b in fl.dr["boxes"][str(f)] if str(b[0]) == tid)
    l_, t_, w_, h_ = box[1:5]
    o = fl.objs[tid]
    base = fl.draw(k, 10 * FPS)
    x, y, bw, bh = VID
    dim = base.copy()
    dim[y : y + bh, x : x + bw] = (dim[y : y + bh, x : x + bw] * 0.45).astype(np.uint8)
    bx0, by0 = int(x + l_ * fl.scale), int(y + t_ * fl.scale)
    bx1, by1 = int(x + (l_ + w_) * fl.scale), int(y + (t_ + h_) * fl.scale)
    pad = 6
    keep = (slice(by0 - pad, by1 + pad), slice(bx0 - pad, bx1 + pad))
    dim[keep] = base[keep]
    cv2.rectangle(dim, (bx0 - 3, by0 - 3), (bx1 + 3, by1 + 3), ORANGE, 2, cv2.LINE_AA)
    # magnified view from the full-resolution frame
    full = fl.frame(k)
    fs = full.shape[1] / float(fl.dr["video"]["width"])  # stored frames may be smaller
    cx, cy = int((l_ + w_ / 2) * fs), int((t_ + h_ / 2) * fs)
    half = int(max(w_, h_) * fs * 1.6) + 10
    crop = full[max(cy - half, 0) : cy + half, max(cx - half, 0) : cx + half]
    zs = 170
    zoom = cv2.resize(crop, (zs, zs), interpolation=cv2.INTER_CUBIC)
    zx = x + 16 if bx0 > x + bw / 2 else x + bw - zs - 16
    zy = y + 16
    dim[zy : zy + zs, zx : zx + zs] = zoom
    cv2.rectangle(dim, (zx - 1, zy - 1), (zx + zs, zy + zs), ORANGE, 2, cv2.LINE_AA)
    near = (zx + zs, zy + zs // 2) if zx < bx0 else (zx, zy + zs // 2)
    cv2.line(dim, near, ((bx0 + bx1) // 2, by0 - 3), ORANGE, 1, cv2.LINE_AA)
    # line to the map and the label there
    mp = tuple(int(v) for v in fl.latest[tid])
    cv2.line(dim, (bx1 + 3, (by0 + by1) // 2), mp, ORANGE, 1, cv2.LINE_AA)
    cv2.circle(dim, mp, 7, ORANGE, -1, cv2.LINE_AA)
    cv2.circle(dim, mp, 12, ORANGE, 2, cv2.LINE_AA)
    rng, a_, b_ = o["range_m"], o["ellipse_major_m"], o["ellipse_minor_m"]
    lines = ["Person", f"{rng:.0f} m from the drone", f"95 % area: {a_:.1f} × {b_:.1f} m"]
    n = int(FREEZE_S * FPS)
    cap_text = CAPTIONS["b"].format(range=f"{rng:.0f}", size=f"{a_:.1f}")  # 95 % ellipse, long axis
    for j in range(n):
        pil = chrome(dim, fl.counts(fl.t(k)))
        d = ImageDraw.Draw(pil)
        lx = min(mp[0] + 18, MAP[0] + MAP[2] - 190)
        ly = max(mp[1] - 70, MAP[1] + 8)
        d.rounded_rectangle((lx, ly, lx + 186, ly + 66), 6, fill=(20, 22, 26), outline=rgb(ORANGE))
        for m, s in enumerate(lines):
            d.text((lx + 10, ly + 6 + m * 19), s, font=F_LEGEND if m else F_TITLE,
                   fill=(240, 240, 240) if m == 0 else rgb(INK2))  # fmt: skip
        yield caption(pil, cap_text, cap_alpha(j, n))


def skip(fl: Flight, k: int) -> Iterator[np.ndarray]:
    """The current frame darkened under a line saying the flight is skipped ahead."""
    base = fl.draw(k, 0)
    n = int(SKIP_S * FPS)
    for j in range(n):
        a = min(1.0, (j + 1) / (0.3 * FPS))
        img = (base * (1 - 0.7 * a)).astype(np.uint8)
        pil = Image.fromarray(img[..., ::-1])
        ImageDraw.Draw(pil).text(
            (W // 2, H // 2), CAPTIONS["d"], font=font(30, True), fill=(240, 240, 240), anchor="mm"
        )
        yield np.array(pil)[..., ::-1].copy()


def ending(fl: Flight, seconds: float, keys: list[str], pulse: bool) -> Iterator[np.ndarray]:
    k = len(fl.vframes) - 1
    fl.advance(fl.vframes[k], 0)
    n = int(seconds * FPS)
    per = n // len(keys)
    c = fl.counts(fl.t(k))
    for j in range(n):
        img = fl.draw(k, j + 100 * FPS, pulse_grey=pulse)
        if pulse:  # the camera view steps back so the map's grey markers lead
            x, y, bw, bh = VID
            img[y : y + bh, x : x + bw] = (img[y : y + bh, x : x + bw] * 0.5).astype(np.uint8)
        pil = chrome(img, c)
        m = min(j // per, len(keys) - 1)
        yield caption(pil, CAPTIONS[keys[m]], cap_alpha(j - m * per, per))


def card() -> Iterator[np.ndarray]:
    n = int(CARD_S * FPS)
    lines = [
        ("Drone Ranger", font(54, True), (245, 245, 245), -40),
        ("From a drone's camera to a map, on a Raspberry Pi", font(24), (215, 215, 215), 25),
        ("Next: a team of drones, and one goes silent", font(20), rgb(ORANGE), 80),
    ]
    for j in range(n):
        a = min(1.0, (j + 1) / (0.5 * FPS))
        pil = Image.new("RGB", (W, H), rgb(BG))
        d = ImageDraw.Draw(pil)
        for s, f, col, dy in lines:
            c = tuple(int(v * a + b * (1 - a)) for v, b in zip(col, rgb(BG), strict=True))
            d.text((W // 2, H // 2 + dy), s, font=f, fill=c, anchor="mm")
        yield np.array(pil)[..., ::-1].copy()


def fade_out(gen: Any, n_total: int, s: float = 0.5) -> Iterator[np.ndarray]:
    """Fade to the background at the end only: the first frame is the still a player shows
    before it is pressed, so it must be a full, labelled frame, not black."""
    nf = int(s * FPS)
    for j, img in enumerate(gen):
        a = min(1.0, (n_total - j) / nf)
        yield img if a >= 1 else (img * a + np.array(BG) * (1 - a)).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", default="cache/maps/syndrone_Town07_h50")
    ap.add_argument("--crf", type=int, default=26)
    args = ap.parse_args(argv)
    folder = REPO_ROOT / args.map
    fl = Flight(folder)
    if fl.label(FREEZE_TRACK) != "real" or fl.objs[FREEZE_TRACK]["class"] != "person":
        raise SystemExit(f"track {FREEZE_TRACK} is not a real person in this map")
    ka, kc = fl.k_at(FLIGHT_A_S), fl.k_at(FLIGHT_C_END_S)
    if not any(str(b[0]) == FREEZE_TRACK for b in fl.dr["boxes"].get(str(fl.vframes[ka]), [])):
        raise SystemExit(f"track {FREEZE_TRACK} has no box at {FLIGHT_A_S} s")
    segments = [
        ("flight start, slowed", play(fl, 0, ka, [(0.0, "a1")], None, OPEN_SPEED)),
        ("freeze on one real person", freeze(fl, ka, FREEZE_TRACK)),
        ("flight on, real speed", play(fl, ka, kc, [(0.0, "a2"), (6.0, "c1"), (12.5, "c2")], 12.5)),
        ("skip to the end", skip(fl, kc)),
        ("whole flight", ending(fl, WHOLE_S, ["e"], False)),
        ("false alarms", ending(fl, FALSE_S, ["f1", "f2"], True)),
        ("end card", card()),
    ]

    def frames() -> Any:
        for _, g in segments:
            yield from g

    flight_s = fl.t(ka) / OPEN_SPEED + (fl.t(kc) - fl.t(ka)) / SPEED
    n_total = int((flight_s + FREEZE_S + SKIP_S + WHOLE_S + FALSE_S + CARD_S) * FPS)
    dest = REPO_ROOT / "cache" / "video" / "demo_one_drone.mp4"
    n = encode(fade_out(frames(), n_total), dest, args.crf)
    mb = dest.stat().st_size / 1e6
    final = fl.counts(fl.t(len(fl.vframes) - 1))
    o = fl.objs[FREEZE_TRACK]
    out = {
        "file": dest.relative_to(REPO_ROOT).as_posix(),
        "seconds": round(n / FPS, 2),
        "size_mb": round(mb, 2),
        "size": [W, H],
        "encoding": f"H.264 crf {args.crf}",
        "limits": {"max_mb": MAX_MB, "max_s": MAX_S},
        "within_limits": mb <= MAX_MB and n / FPS <= MAX_S,
        "source": folder.relative_to(REPO_ROOT).as_posix(),
        "flight": fl.dr.get("title"),
        "split": "development",
        "speed": SPEED,
        "opening_speed": OPEN_SPEED,
        "flight_shown_s": [0.0, round(fl.t(kc), 2)],
        "frozen_person": {
            "track_id": FREEZE_TRACK,
            "at_s": round(fl.t(ka), 2),
            "range_m": round(o["range_m"], 1),
            "ellipse_95_m": [round(o["ellipse_major_m"], 2), round(o["ellipse_minor_m"], 2)],
        },
        "segments": [name for name, _ in segments],
        "captions": CAPTIONS,
        "final_counts": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()},
        "searched_km2_map": fl.dr["searched_km2"],
    }
    rv = REPO_ROOT / "results" / "video" / "demo_one_drone.json"
    rv.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("file", "seconds", "size_mb", "within_limits")}))
    return 0 if out["within_limits"] else 1


if __name__ == "__main__":
    sys.exit(main())
