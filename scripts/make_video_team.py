"""Video 2, a team of drones: one goes silent and the others take over its ground.

Usage: python scripts/make_video_team.py [--crf 24] [--loop docs/media/team_lost_drone.webp]

Made, like video 1 (make_video_one_drone.py), to be understood with the sound off by someone
who knows nothing about the project: captions, a counter, a roster of the drones and a legend.

Source: two searches of the frozen Phase 16 simulation on the same missing person, the same
camera and the same lost drone, recorded by make_demo_v2.record (which wraps the engine without
changing it): the sweep planner on the frozen settings ("the team plan") and the lawnmower
("fixed lanes": each drone keeps its own strip). The seed is the one make_demo_v2.choose_seed
chose on development seeds by its written rule (results/video/demo_v2_seed_choice.json):
the typical case among those where the person is on ground the lost drone had not yet searched.

Sequence (25 frames per second; the simulated time per second of video is in SCHEDULE):
1. four drones fly their lanes, up to the moment one is lost (the video opens START_S into
   the search, so that its first frame, the still a player shows, already has lanes on it);
2. the lost drone turns grey, the ground it left unsearched is hatched, and the seconds of
   silence are counted until the others presume it lost (the frozen lost_after_s);
3. the others finish their own lanes, then fly the hatched ground;
4. the person is seen there, then confirmed by a second sighting at the same place;
5. side by side: the team plan's finished search and the fixed lanes' whole search, where the
   hatched ground is never flown and the person is never seen;
6. an end card with the test result this case illustrates, how often such a case arises, and
   the open problem (places reported to the crew that were not the person).

Drawn: the ground each drone covered first, in its colour (dark: not yet covered); each drone's
path; the drones as numbered discs (1 to 4: the engine's index plus one); the missing person
from the start, as an orange dot (the drones do not know it). Not drawn, unlike the earlier
demo: single detections and the places reported to the crew, which are counted in a tile
instead, so that the takeover stays readable.

Writes cache/video/demo_team.mp4 (not committed), results/video/demo_team.json (seed, schedule,
captions, outcomes, the numbers on the end card with their sources, size, limits) and, with
--loop, a short silent loop of the map panel for the README.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerialrange.utils.config import REPO_ROOT  # noqa: E402
from make_demo_v2 import (  # noqa: E402
    BG,
    DRONES,
    FPS,
    GREY,
    ORANGE,
    H,
    W,
    choose_seed,
    configs16,
    encode,
    record,
)
from make_video_one_drone import (  # noqa: E402
    CAPTION_FADE_S,
    INK2,
    TILE,
    cap_alpha,
    caption,
    fade_out,
    fmt_time,
    font,
    rgb,
)

MAX_MB, MAX_S = 10.0, 60.0  # GitHub plays an uploaded video up to 10 MB
UNSEARCHED = (30, 26, 22)  # BGR, the map before any drone has covered it
SHADE = 0.62  # weight of a drone's colour on ground it covered
SIDE = 528  # map panel, pixels (2.4 per 10 m cell)
ONE = (56, 100)  # the single map's top-left corner
PAIR = ((64, 122), (W - 64 - SIDE, 122))  # the two maps of the comparison
ROSTER_X = ONE[0] + SIDE + 48
HOLD_FOUND_S, PAIR_HOLD_S, CARD_S = 3.5, 4.5, 6.0
START_S = 90.0  # simulated seconds already flown when the video opens
# (name, from event, to event, video seconds, captions)
SCHEDULE = [
    ("search", "start", "lost", 9.0, ["a1", "a2"]),
    ("silence", "lost", "presumed", 5.5, ["b"]),
    ("own lanes", "presumed", "takeover", 6.5, ["c"]),
    ("takeover", "takeover", "seen", 3.5, ["d1"]),
    ("find", "seen", "confirmed", 3.5, ["d2"]),
]
PAIR_PLAY_S = 7.0
F_LABEL, F_VALUE, F_TITLE = font(13, True), font(28, True), font(16, True)
F_BODY, F_NUM, F_TAG = font(15), font(14, True), font(14, True)

CAPTIONS = {
    "a1": "Four drones search {km2:.1f} km² for one missing person.\n"
    "Each colour is the ground one drone has covered.",
    "a2": "They fly in lanes and tell each other what they have covered,\n"
    "in a few dozen bytes a second.",
    "b": "Drone {lost} goes silent: a flat battery, a crash or a lost radio.\n"
    "The others cannot tell which.",
    "c": "After {after:.0f} seconds of silence they count it as lost,\n"
    "and add the ground it left to their own.",
    "d1": "Their own lanes finished, they fly the ground the lost drone left.",
    "d2": "Two sightings at the same spot: the person is confirmed and a crew is sent.",
    "e1": "The same search without the team plan: each drone keeps to its own strip.",
    "e2": "The lost drone's ground is never searched, and the person was in it.",
}


class Search:
    """One recorded search, with what the drawing needs worked out once."""

    def __init__(self, r: dict[str, Any], lost_after_s: float) -> None:
        g = self.g = r["grid"]
        self.dt = float(r["dt"])
        self.pos, self.res = r["pos"], r["result"]
        self.n = self.pos.shape[1]
        self.end = len(self.pos) - 1
        self.first = r["first"].reshape(g.nrows, g.ncols)[::-1]
        own = r["owner"].reshape(g.nrows, g.ncols)[::-1]
        self.colour = np.array(DRONES, np.uint8)[np.maximum(own, 0)]
        self.lost, self.k_lost = (int(v) for v in r["lose"])
        self.k_presumed = self.k_lost + int(round(lost_after_s / self.dt))
        # the lost drone's strip (as make_demo_v2.lost_drone_ground) not covered when it stopped
        w = g.ncols / self.n
        strip = np.zeros((g.nrows, g.ncols), bool)
        strip[:, int(round(self.lost * w)) : int(round((self.lost + 1) * w))] = True
        self.left = strip & ((self.first < 0) | (self.first > self.k_lost))
        later = self.first[self.left & (self.first > self.k_lost)]
        self.k_takeover = int(later.min()) if len(later) else None
        self.k_seen = None
        if self.res.t_found_s is not None:
            self.k_seen = int(round(self.res.t_found_s / self.dt)) - 1
        self.alarm_k = np.array([k for _, _, k, ok in r["alarms"] if not ok], int)
        self.person = r["person"]
        self.km2 = g.nrows * g.ncols * g.cell_m**2 / 1e6

    def marks(self) -> dict[str, int]:
        return {
            "start": int(round(START_S / self.dt)),
            "lost": self.k_lost,
            "presumed": self.k_presumed,
            "takeover": self.k_takeover,
            "seen": self.k_seen,
            "confirmed": self.end,
        }

    def t(self, k: int) -> float:
        """Simulated seconds at recorded step k (the drones' positions before step k + 1)."""
        return k * self.dt

    def searched(self, k: int) -> float:
        return float(((self.first >= 0) & (self.first <= k)).mean())

    def reported(self, k: int) -> int:
        return int(np.searchsorted(self.alarm_k, k, side="right"))

    def px(self, n: float, e: float, box: tuple[int, int]) -> tuple[int, int]:
        g, s = self.g, SIDE / self.g.ncols
        return (
            int(round(box[0] + (e - g.e0_m) / g.cell_m * s)),
            int(round(box[1] + SIDE - (n - g.n0_m) / g.cell_m * s)),
        )

    def in_left(self, k: int, i: int) -> bool:
        """Whether drone i is over the ground the lost drone left, after it was lost."""
        if k <= self.k_lost or i == self.lost:
            return False
        g = self.g
        n_, e_ = self.pos[k][i]
        c = int((e_ - g.e0_m) // g.cell_m)
        r = g.nrows - 1 - int((n_ - g.n0_m) // g.cell_m)
        return 0 <= r < g.nrows and 0 <= c < g.ncols and bool(self.left[r, c])


def draw_map(
    img: np.ndarray, s: Search, k: int, box: tuple[int, int], j: int, never: bool = False
) -> list[tuple[int, int, str, str]]:
    """The map at step k, drawn on img; returns the text to add: (x, y, text, kind)."""
    x0, y0 = box
    k = min(k, s.end)
    cov = (s.first >= 0) & (s.first <= k)
    tile = np.full((*cov.shape, 3), UNSEARCHED, np.uint8)
    tile[cov] = (SHADE * s.colour[cov] + (1 - SHADE) * np.array(UNSEARCHED)).astype(np.uint8)
    win = img[y0 : y0 + SIDE, x0 : x0 + SIDE]
    win[:] = cv2.resize(tile, (SIDE, SIDE), interpolation=cv2.INTER_LINEAR)
    text: list[tuple[int, int, str, str]] = []
    if k >= s.k_lost:  # the ground the lost drone left, hatched until another drone covers it
        rest = cv2.morphologyEx(  # without the odd cells a pass leaves between lanes
            (s.left & ~cov).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        m = cv2.resize(rest, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST) > 0
        yy, xx = np.nonzero(m)
        if len(yy):
            a = 0.55 + 0.25 * np.sin(j / FPS * 2 * np.pi * 0.9)
            stripe = ((xx + yy) // 5) % 2 == 0
            px_ = win[yy[stripe], xx[stripe]]
            win[yy[stripe], xx[stripe]] = (px_ * (1 - a) + 235 * a).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(win, cs, -1, (240, 240, 240), 2, cv2.LINE_AA)
    for i, c in enumerate(DRONES[: s.n]):  # paths
        kk = min(k, s.k_lost) if i == s.lost else k
        tr = np.array([s.px(*p, box) for p in s.pos[: kk + 1, i]], np.int32)
        if len(tr) > 1:
            light = tuple(int(0.55 * v + 0.45 * 255) for v in c)
            cv2.polylines(img, [tr], False, light, 1, cv2.LINE_AA)
    lx, ly = s.px(*s.pos[0][0], box)  # launch point
    cv2.drawMarker(img, (lx, ly), (245, 245, 245), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    for i, c in enumerate(DRONES[: s.n]):  # drones, the lost one last so it stays on top
        if i == s.lost and k > s.k_lost:
            continue
        cx, cy = s.px(*s.pos[k][i], box)
        cv2.circle(img, (cx, cy), 12, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 10, c, -1, cv2.LINE_AA)
        text.append((cx, cy, str(i + 1), "num"))
    if k > s.k_lost:
        cx, cy = s.px(*s.pos[s.k_lost][s.lost], box)
        age = (k - s.k_lost) * s.dt
        if age < 90:  # warning rings while the others have not yet reacted
            for ph in (0, 15):
                cv2.circle(img, (cx, cy), 14 + (j + ph) % 30, (70, 70, 240), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 12, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 10, GREY, -1, cv2.LINE_AA)
        cv2.line(img, (cx - 5, cy - 5), (cx + 5, cy + 5), (30, 30, 30), 2, cv2.LINE_AA)
        cv2.line(img, (cx - 5, cy + 5), (cx + 5, cy - 5), (30, 30, 30), 2, cv2.LINE_AA)
        text.append((cx, cy, f"Drone {s.lost + 1} silent", "lost"))
    q = s.px(*s.person, box)  # the person, over the drones so a find is never hidden
    seen = s.k_seen is not None and k >= s.k_seen
    done = s.res.confirmed and k >= s.end
    if seen:  # no drone's number printed across the find
        text = [t for t in text if t[3] != "num" or max(abs(t[0] - q[0]), abs(t[1] - q[1])) > 13]
    cv2.circle(img, q, 7, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, q, 5, ORANGE, -1, cv2.LINE_AA)
    if done:
        for ph in (0, 12):
            cv2.circle(img, q, 10 + (j + ph) % 24, ORANGE, 2, cv2.LINE_AA)
        text.append((q[0], q[1], "Confirmed: crew sent", "found"))
    elif seen:
        cv2.circle(img, q, 12, ORANGE, 2, cv2.LINE_AA)
        text.append((q[0], q[1], "Seen once", "found"))
    elif never:
        cv2.circle(img, q, 12, (110, 110, 240), 2, cv2.LINE_AA)
        text.append((q[0], q[1], "Never seen", "lost"))
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + SIDE, y0 + SIDE), (80, 76, 72), 1)
    return text


def draw_text(d: ImageDraw.ImageDraw, text: list, box: tuple[int, int]) -> None:
    for x, y, s, kind in text:
        if kind == "num":
            d.text((x, y), s, font=F_NUM, fill=(20, 20, 20), anchor="mm")
            continue
        col = rgb(ORANGE) if kind == "found" else (240, 110, 110)
        w = int(d.textlength(s, font=F_TAG)) + 16
        left = x + 18 if x + 18 + w < box[0] + SIDE else x - 18 - w
        top = max(y - 34, box[1] + 4)
        d.rounded_rectangle((left, top, left + w, top + 24), 5, fill=(20, 22, 26), outline=col)
        d.text((left + 8, top + 12), s, font=F_TAG, fill=(240, 240, 240), anchor="lm")


def tiles(d: ImageDraw.ImageDraw, items: list[tuple[str, str]], warn: int | None) -> None:
    gap, x0 = 12, 24
    tw = (W - 48 - (len(items) - 1) * gap) // len(items)
    for n, (lab, val) in enumerate(items):
        x = x0 + n * (tw + gap)
        d.rounded_rectangle((x, 12, x + tw, 74), 8, fill=rgb(TILE))
        if n == warn:
            d.rounded_rectangle((x, 12, x + tw, 74), 8, outline=(240, 110, 110), width=3)
        d.text((x + 14, 20), lab, font=F_LABEL, fill=rgb(INK2))
        d.text((x + 14, 34), val, font=F_VALUE, fill=(245, 245, 245))


def roster(d: ImageDraw.ImageDraw, s: Search, k: int) -> None:
    x, y = ROSTER_X, ONE[1]
    d.text((x, y), "The team", font=F_TITLE, fill=(230, 230, 230))
    y += 36
    for i, c in enumerate(DRONES[: s.n]):
        gone = i == s.lost and k > s.k_lost
        d.ellipse((x, y, x + 24, y + 24), fill=rgb(GREY) if gone else rgb(c))
        d.text((x + 12, y + 12), str(i + 1), font=F_NUM, fill=(20, 20, 20), anchor="mm")
        if not gone:
            status, col = "searching its lanes", rgb(INK2)
            if s.in_left(k, i):
                status, col = f"searching drone {s.lost + 1}'s ground", (240, 240, 240)
            if s.res.confirmed and k >= s.end:
                status = "search over"
        elif k < s.k_presumed:
            status, col = f"silent for {(k - s.k_lost) * s.dt:.0f} s", (240, 110, 110)
        else:
            status, col = "counted as lost by the others", (240, 110, 110)
        d.text((x + 36, y + 12), f"Drone {i + 1}", font=F_TITLE, fill=(235, 235, 235), anchor="lm")
        d.text((x + 112, y + 12), status, font=F_BODY, fill=col, anchor="lm")
        y += 40
    y += 26
    d.text((x, y), "On the map", font=F_TITLE, fill=(230, 230, 230))
    y += 34
    rows = [
        ("dark", "not searched yet"),
        ("colour", "searched, in the colour of the drone that did"),
        ("hatch", "left unsearched by the lost drone"),
        ("person", "the missing person (the drones do not know)"),
        ("launch", "launch point"),
    ]
    for kind, name in rows:
        if kind == "dark":
            d.rectangle((x, y + 2, x + 20, y + 18), fill=rgb(UNSEARCHED), outline=(90, 90, 90))
        elif kind == "colour":
            for m, c in enumerate(DRONES[: s.n]):
                col = tuple(
                    int(SHADE * v + (1 - SHADE) * u) for v, u in zip(c, UNSEARCHED, strict=True)
                )
                d.rectangle((x + m * 5, y + 2, x + m * 5 + 5, y + 18), fill=rgb(col))
        elif kind == "hatch":
            d.rectangle((x, y + 2, x + 20, y + 18), fill=rgb(UNSEARCHED), outline=(240, 240, 240))
            for m in range(-12, 20, 6):
                d.line((x + max(m, 0), y + 18 - max(-m, 0), x + min(m + 16, 20),
                        y + 2 + max(m + 16 - 20, 0)), fill=(235, 235, 235), width=2)  # fmt: skip
        elif kind == "person":
            d.ellipse((x + 4, y + 4, x + 16, y + 16), fill=rgb(ORANGE), outline=(20, 20, 20))
        else:
            d.line((x + 3, y + 10, x + 17, y + 10), fill=(245, 245, 245), width=2)
            d.line((x + 10, y + 3, x + 10, y + 17), fill=(245, 245, 245), width=2)
        d.text((x + 34, y + 10), name, font=F_BODY, fill=rgb(INK2), anchor="lm")
        y += 30


def one_frame(s: Search, k: int, j: int) -> Image.Image:
    img = np.full((H, W, 3), BG, np.uint8)
    text = draw_map(img, s, k, ONE, j)
    pil = Image.fromarray(img[..., ::-1])
    d = ImageDraw.Draw(pil)
    draw_text(d, text, ONE)
    flying = s.n - (1 if k > s.k_lost else 0)
    tiles(
        d,
        [
            ("SEARCH TIME", fmt_time(s.t(k))),
            ("GROUND SEARCHED", f"{s.searched(k):.0%}"),
            ("DRONES FLYING", f"{flying} of {s.n}"),
            ("PLACES REPORTED TO THE CREW", f"{s.reported(k):,}"),
        ],
        2 if k > s.k_lost else None,
    )
    roster(d, s, k)
    return pil


def single(s: Search, fmt: dict) -> Iterator[np.ndarray]:
    """The team plan's search from START_S to the confirmed find, then held."""
    marks = s.marks()
    j = 0
    for name, a, b, secs, keys in SCHEDULE:
        k0, k1 = marks[a], marks[b]
        n = int(secs * FPS)
        hold = int(HOLD_FOUND_S * FPS) if name == SCHEDULE[-1][0] else 0
        per = (n + hold) // len(keys)
        for f in range(n + hold):
            k = k1 if f >= n else int(round(k0 + (k1 - k0) * f / n))
            m = min(f // per, len(keys) - 1)
            j += 1
            text = CAPTIONS[keys[m]].format(**fmt)
            # no fade-in on the opening caption: the first frame is the player's still
            alpha = 1.0 if j <= CAPTION_FADE_S * FPS else cap_alpha(f - m * per, per)
            yield caption(one_frame(s, k, j), text, alpha)


def pair(team: Search, fixed: Search, fmt: dict) -> Iterator[np.ndarray]:
    """Left: the team plan's finished search. Right: the fixed lanes' whole search, played."""
    n, hold = int(PAIR_PLAY_S * FPS), int(PAIR_HOLD_S * FPS)
    for f in range(n + hold):
        k = fixed.end if f >= n else int(round(fixed.end * f / n))
        img = np.full((H, W, 3), BG, np.uint8)
        t_left = draw_map(img, team, team.end, PAIR[0], f)
        t_right = draw_map(img, fixed, k, PAIR[1], f, never=f >= n)
        pil = Image.fromarray(img[..., ::-1])
        d = ImageDraw.Draw(pil)
        draw_text(d, t_left, PAIR[0])
        draw_text(d, t_right, PAIR[1])
        heads = ("With the team plan", "Without it: each drone keeps to its own strip")
        found = f"Person confirmed after {fmt_time(team.t(team.end))}"
        if f >= n:
            state = f"Not found after {fmt_time(fixed.t(k))}: search called off"
            col = (240, 110, 110)
        else:
            state, col = f"Searching: {fmt_time(fixed.t(k))}", rgb(INK2)
        rows = zip(PAIR, heads, (found, state), (rgb(ORANGE), col), strict=True)
        for (x, y), head, line, c in rows:
            d.text((x, y - 98), head, font=font(22, True), fill=(240, 240, 240))
            d.text((x, y - 62), line, font=font(18), fill=c)
            lost = f"Drone {team.lost + 1} lost after {fmt_time(team.t(team.k_lost + 1))}"
            d.text((x, y - 32), lost, font=F_BODY, fill=rgb(INK2))
        key = "e1" if f < n * 0.55 else "e2"
        f0, n0 = (0, int(n * 0.55)) if key == "e1" else (int(n * 0.55), n + hold - int(n * 0.55))
        yield caption(pil, CAPTIONS[key].format(**fmt), cap_alpha(f - f0, n0))


def card(lines: list[tuple[str, Any, tuple[int, int, int], int]]) -> Iterator[np.ndarray]:
    n = int(CARD_S * FPS)
    for j in range(n):
        a = min(1.0, (j + 1) / (0.5 * FPS))
        pil = Image.new("RGB", (W, H), rgb(BG))
        d = ImageDraw.Draw(pil)
        for s, f, col, dy in lines:
            c = tuple(int(v * a + b * (1 - a)) for v, b in zip(col, rgb(BG), strict=True))
            d.multiline_text((W // 2, H // 2 + dy), s, font=f, fill=c, anchor="mm", align="center",
                             spacing=8)  # fmt: skip
        yield np.array(pil)[..., ::-1].copy()


def write_loop(src: Path, dest: Path, t0: float, t1: float) -> None:
    """A silent loop of the single map and its roster, for the README."""
    x, y, w, h = ONE[0] - 16, ONE[1] - 12, W - ONE[0] - 8, SIDE + 24
    vf = f"crop={w}:{h}:{x}:{y},fps=10,setpts=PTS/1.5,scale=880:-2:flags=lanczos"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0}", "-to", f"{t1}", "-i", str(src),
           "-vf", vf, "-loop", "0", "-c:v", "libwebp", "-quality", "70", "-compression_level", "6",
           "-an", str(dest)]  # fmt: skip
    dest.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(cmd).returncode:
        raise SystemExit("ffmpeg failed writing the loop")


def main(argv: list[str] | None = None) -> int:
    import multidrone16_run as r16

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=5, help="for the seed choice, if not on file")
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--loop", default=None, help="also write a README loop of the map (webp)")
    args = ap.parse_args(argv)
    choice = choose_seed(args.workers)
    cfgs = configs16()
    after = float(cfgs["sweep"]["pc"]["lost_after_s"])
    team = Search(record(cfgs["sweep"], choice["seed"]), after)
    fixed = Search(record(cfgs["lawnmower"], choice["seed"]), after)
    if team.res.t_confirmed_s != choice["sweep_t_s"] or fixed.res.confirmed:
        raise SystemExit("the recorded searches differ from the seed-choice run")
    if team.k_takeover is None or team.k_seen is None or not team.k_takeover <= team.k_seen:
        raise SystemExit("this seed does not show a takeover before the find")
    paired = json.loads(
        (REPO_ROOT / "results/multi_drone/phase16_paired_test.json").read_text(encoding="utf-8")
    )["comparisons"]
    cost = {
        k: paired[f"{k}_lose_any_minus_no_loss"]["time"]["mean"] for k in ("sweep", "lawnmower")
    }
    fmt = {"km2": team.km2, "lost": team.lost + 1, "after": after}
    end_lines = [
        (f"Over {len(r16.TEST_SEEDS)} test searches, losing a drone added "
         f"{cost['sweep']:,.0f} s\nwith the team plan and {cost['lawnmower']:,.0f} s without.",
         font(34, True), (245, 245, 245), -110),
        (f"A case like this one, with the person on the lost drone's ground, came up in "
         f"{choice['n_candidates']} of {choice['n_seeds']} trial searches.\n"
         "With no drone lost, neither plan was measurably faster.",
         font(21), (215, 215, 215), 5),
        (f"Still open: false alarms. Here {team.res.crew_alarms_before_confirmed:,} places were "
         "reported to the crew before the right one.",
         font(21), rgb(ORANGE), 95),
        ("Drone Ranger  ·  simulation, camera measured on rendered towns", font(17),
         rgb(INK2), 170),
    ]  # fmt: skip
    segments = [
        ("the team plan, to the confirmed find", single(team, fmt)),
        ("side by side with fixed lanes", pair(team, fixed, fmt)),
        ("end card", card(end_lines)),
    ]

    def frames() -> Any:
        for _, g in segments:
            yield from g

    single_s = sum(row[3] for row in SCHEDULE) + HOLD_FOUND_S
    total_s = single_s + PAIR_PLAY_S + PAIR_HOLD_S + CARD_S
    dest = REPO_ROOT / "cache" / "video" / "demo_team.mp4"
    n = encode(fade_out(frames(), int(total_s * FPS)), dest, args.crf)
    mb = dest.stat().st_size / 1e6
    marks = team.marks()
    out = {
        "file": dest.relative_to(REPO_ROOT).as_posix(),
        "seconds": round(n / FPS, 2),
        "size_mb": round(mb, 2),
        "size": [W, H],
        "encoding": f"H.264 crf {args.crf}",
        "limits": {"max_mb": MAX_MB, "max_s": MAX_S},
        "within_limits": mb <= MAX_MB and n / FPS <= MAX_S,
        "source": "two searches of the frozen Phase 16 simulation (configs/search16.yaml), "
        "recorded by scripts/make_demo_v2.py record(): the sweep planner and the lawnmower, four "
        "drones, primary camera, perfect radio, the same drone lost at half a sweep's time",
        "seed": choice["seed"],
        "seed_choice_file": "results/video/demo_v2_seed_choice.json",
        "split": "development seeds",
        "lost_drone_shown_as": team.lost + 1,
        # the loss falls within step k_lost, so it and the timeout count from that step's end
        "events_s": {k: team.t(v + (k in ("lost", "presumed"))) for k, v in marks.items()},
        "schedule": [
            *({"segment": name, "video_s": secs} for name, _, _, secs, _ in SCHEDULE),
            {"segment": "find held", "video_s": HOLD_FOUND_S},
            {"segment": "side by side", "video_s": PAIR_PLAY_S + PAIR_HOLD_S},
            {"segment": "end card", "video_s": CARD_S},
        ],
        "team_plan": {
            "confirmed": team.res.confirmed,
            "t_confirmed_s": team.res.t_confirmed_s,
            "places_reported_before_find": team.res.crew_alarms_before_confirmed,
        },
        "fixed_lanes": {
            "confirmed": fixed.res.confirmed,
            "searched_until_s": fixed.t(fixed.end),
            "ground_searched_share": round(fixed.searched(fixed.end), 4),
        },
        "end_card": {
            "test_searches": len(r16.TEST_SEEDS),
            "added_s_team_plan": cost["sweep"],
            "added_s_fixed_lanes": cost["lawnmower"],
            "source": "results/multi_drone/phase16_paired_test.json",
            "cases_like_this": [choice["n_candidates"], choice["n_seeds"]],
        },
        "not_drawn": "single detections and places reported to the crew (counted in a tile)",
        "captions": {k: v.format(**fmt) for k, v in CAPTIONS.items()},
        "end_card_lines": [s for s, *_ in end_lines],
    }
    if args.loop:
        loop = REPO_ROOT / args.loop
        t0 = SCHEDULE[0][3] - 2.0
        write_loop(dest, loop, t0, single_s)
        out["loop"] = {
            "file": loop.relative_to(REPO_ROOT).as_posix(),
            "from_s": t0,
            "to_s": single_s,
            "size_mb": round(loop.stat().st_size / 1e6, 2),
        }
    rv = REPO_ROOT / "results" / "video" / "demo_team.json"
    rv.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("file", "seconds", "size_mb", "within_limits")}))
    return 0 if out["within_limits"] else 1


if __name__ == "__main__":
    sys.exit(main())
