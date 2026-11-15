"""Shared pieces for the 3D field-view video (scripts/make_video_field_view.py).

Flight data from a search map, a ground texture revealed as it is searched, a small 3D camera
(ground-plane homography and point projection), fonts, easing and text helpers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from make_demo_v2 import load_map  # noqa: E402

W, H, FPS = 1280, 720, 25
M_PER_DEG = 111_320.0  # the maps' origin is latitude 0, so one scale serves both axes
ORANGE, TEAL, GREY = (40, 150, 255), (190, 200, 60), (150, 150, 150)  # BGR
RED = (90, 90, 240)
DRONES = [(255, 170, 60), (90, 220, 120), (230, 110, 230), (60, 220, 230)]  # BGR


def rgb(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (c[2], c[1], c[0])


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    names = {
        "regular": "segoeui.ttf",
        "semibold": "seguisb.ttf",
        "bold": "segoeuib.ttf",
        "black": "seguibl.ttf",
        "light": "segoeuil.ttf",
    }
    p = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / names[weight]
    if not p.exists():
        import matplotlib

        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        p = p / ("DejaVuSans.ttf" if weight in ("regular", "light") else "DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(p), size)


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a: Any, b: Any, x: float) -> Any:
    return a + (b - a) * x


def fade(f: int, n: int, secs: float = 0.3) -> float:
    """Alpha of something shown for n frames, at frame f: in and out over `secs`."""
    k = secs * FPS
    return float(min(1.0, (f + 1) / k, (n - f) / k))


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img[..., ::-1])


def to_bgr(pil: Image.Image) -> np.ndarray:
    return np.array(pil)[..., ::-1].copy()


def shadow_text(
    d: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    f: Any,
    fill: tuple,
    anchor: str = "la",
    alpha: float = 1.0,
    bg: tuple = (10, 14, 18),
    spacing: int = 6,
    align: str = "left",
) -> None:
    """Text with a soft dark edge so it reads over footage; alpha blends towards bg."""
    col = tuple(int(v * alpha + b * (1 - alpha)) for v, b in zip(fill, bg, strict=True))
    d.multiline_text(
        xy,
        text,
        font=f,
        fill=col,
        anchor=anchor,
        spacing=spacing,
        align=align,
        stroke_width=2 if alpha > 0.4 else 0,
        stroke_fill=(8, 10, 14),
    )


def vgrad(top: tuple, bottom: tuple) -> np.ndarray:
    t = np.linspace(0, 1, H)[:, None, None]
    col = (1 - t) * np.array(top, float) + t * np.array(bottom, float)
    return np.broadcast_to(col, (H, W, 3)).astype(np.uint8).copy()


def blend_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, col: tuple, a: float) -> None:
    roi = img[max(y0, 0) : y1, max(x0, 0) : x1]
    roi[:] = (roi * (1 - a) + np.array(col) * a).astype(np.uint8)


def bottom_shade(img: np.ndarray, height: int = 220, strength: float = 0.85) -> None:
    a = (np.linspace(0, 1, height) ** 1.5 * strength)[:, None, None]
    roi = img[H - height :]
    roi[:] = (roi * (1 - a) + np.array((12, 10, 8)) * a).astype(np.uint8)


def top_shade(img: np.ndarray, height: int = 120, strength: float = 0.7) -> None:
    a = (np.linspace(1, 0, height) ** 1.5 * strength)[:, None, None]
    roi = img[:height]
    roi[:] = (roi * (1 - a) + np.array((12, 10, 8)) * a).astype(np.uint8)


# ------------------------------------------------------------------ flight data


class Flight:
    """A search map's data: frames, boxes, detections, objects, footprints, the drone's path."""

    def __init__(self, folder: Path) -> None:
        data = load_map(folder)
        dr = self.dr = data["drone_ranger"]
        self.folder = folder
        self.match = dr["truth_match"]
        self.vframes = [int(f) for f in dr["video"]["frames"]]
        self.t_of = {int(f): float(t) for f, t in dr["timeline"]}
        self.frames = [int(f) for f, _ in dr["timeline"]]
        self.idx = {f: i for i, f in enumerate(self.frames)}
        self.drone_ll = np.array(dr["drone_track"], float)
        self.drone_m = self.drone_ll * M_PER_DEG
        self.box_scale = 1.0 / float(dr["video"]["width"])  # boxes are in full-size pixels
        self.foot: dict[int, np.ndarray] = {}
        self.dets: dict[int, list[dict]] = {}
        self.objs: dict[str, dict] = {}
        for ft in data["features"]:
            p = ft["properties"]
            if p["kind"] == "footprint":
                self.foot[int(p["frame"])] = np.array(ft["geometry"]["coordinates"][0], float)
            elif p["kind"] == "detection":
                q = dict(p, ll=ft["geometry"]["coordinates"], track_id=str(p["track_id"]))
                self.dets.setdefault(int(p["frame"]), []).append(q)
            elif p["kind"] == "object":
                self.objs[str(p["track_id"])] = dict(p, ll=ft["geometry"]["coordinates"])
        self.det_of = {
            (f, d["track_id"]): d for f, ds in self.dets.items() for d in ds
        }  # (frame, track) -> detection
        self.cap = cv2.VideoCapture(str(folder / "video.mp4"))
        self.cur: list = [-1, None]
        # share of the finally searched ground covered by each pipeline frame
        gi = dr["ground_image"]
        self.lon0, self.lat0 = gi["tl"]
        self.dlon, self.dlat = gi["tr"][0] - self.lon0, self.lat0 - gi["bl"][1]
        m = np.zeros((440, 512), np.uint8)
        cum = []
        for f in self.frames:
            if f in self.foot:
                cv2.fillPoly(m, [np.round(self.tex_px(self.foot[f], 512, 440)).astype(np.int32)], 1)
            cum.append(int(m.sum()))
        self.share = np.array(cum, float) / max(cum[-1], 1)

    # geometry
    def tex_px(self, ll: Any, w: int, h: int) -> np.ndarray:
        ll = np.asarray(ll, float).reshape(-1, 2)
        return np.column_stack(
            [(ll[:, 0] - self.lon0) / self.dlon * w, (self.lat0 - ll[:, 1]) / self.dlat * h]
        )

    def tex_to_world(self, w: int, h: int) -> np.ndarray:
        """3x3: texture pixel (u, v, 1) -> ground metres (east, north, 1)."""
        return np.array(
            [
                [self.dlon / w * M_PER_DEG, 0, self.lon0 * M_PER_DEG],
                [0, -self.dlat / h * M_PER_DEG, self.lat0 * M_PER_DEG],
                [0, 0, 1],
            ]
        )

    # time
    def t(self, k: int) -> float:
        return self.t_of[self.vframes[k]]

    def k_at(self, t_s: float) -> int:
        fps = float(self.dr["video"]["fps"])
        return int(min(max(round(t_s * fps), 0), len(self.vframes) - 1))

    def frame(self, k: int) -> np.ndarray:
        if k < self.cur[0]:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            self.cur[0] = k - 1
        while self.cur[0] < k:
            ok, fr = self.cap.read()
            if not ok:
                raise ValueError(f"video ends before frame {k}")
            self.cur[0], self.cur[1] = self.cur[0] + 1, fr
        return self.cur[1]

    # truth and counts
    def label(self, tid: str) -> str:
        return self.match.get(tid, "not scored")

    def colour(self, tid: str) -> tuple[int, int, int]:
        if self.label(tid) in ("real", "near"):
            return ORANGE if self.objs.get(tid, {}).get("class") == "person" else TEAL
        return GREY

    def counts(self, t_s: float) -> dict[str, Any]:
        seen = [o for o in self.objs.values() if o["first_t_s"] <= t_s + 1e-9]
        people = [str(o["track_id"]) for o in seen if o["class"] == "person"]
        i = int(min(max(round(t_s * 25), 0), len(self.share) - 1))
        lab = [self.label(t) for t in people]
        return {
            "people": len(people),
            "vehicles": len(seen) - len(people),
            "km2": self.dr["searched_km2"] * float(self.share[i]),
            "t_s": t_s,
            "false": lab.count("false alarm"),
        }


class GroundTex:
    """The ground image, bright where the flight has searched so far, dim elsewhere.

    Two sizes are kept (full and quarter) so a far view can sample the small one without shimmer.
    """

    def __init__(self, fl: Flight, dim: float = 0.16, base: tuple = (34, 30, 26)) -> None:
        g = cv2.imread(str(fl.folder / "ground.webp"), cv2.IMREAD_UNCHANGED)
        self.fl = fl
        self.levels = []
        for s in (1, 4):
            gs = (
                g
                if s == 1
                else cv2.resize(g, (g.shape[1] // s, g.shape[0] // s), interpolation=cv2.INTER_AREA)
            )
            bright, alpha = gs[..., :3].copy(), gs[..., 3] > 0
            now = (dim * bright + (1 - dim) * np.array(base)).astype(np.uint8)
            now[~alpha] = 0
            mask = np.zeros(alpha.shape, np.uint8)
            self.levels.append({"bright": bright, "alpha": alpha, "now": now, "mask": mask})
        self.done = 0  # pipeline frames applied

    def advance(self, frame: int) -> None:
        fl = self.fl
        while self.done < frame:
            self.done += 1
            ring = fl.foot.get(self.done)
            if ring is None:
                continue
            for lv in self.levels:
                h, w = lv["mask"].shape
                poly = np.round(fl.tex_px(ring, w, h)).astype(np.int32)
                x0, y0 = np.maximum(poly.min(0), 0)
                x1, y1 = np.minimum(poly.max(0) + 1, [w, h])
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(sub, [poly - [x0, y0]], 1)
                new = (sub > 0) & (lv["mask"][y0:y1, x0:x1] == 0) & lv["alpha"][y0:y1, x0:x1]
                lv["mask"][y0:y1, x0:x1][sub > 0] = 1
                lv["now"][y0:y1, x0:x1][new] = lv["bright"][y0:y1, x0:x1][new]

    def level(self, coarse: bool) -> tuple[np.ndarray, np.ndarray]:
        lv = self.levels[1 if coarse else 0]
        h, w = lv["mask"].shape
        return lv["now"], self.fl.tex_to_world(w, h)


# ------------------------------------------------------------------ 3D camera


class Cam:
    """A pinhole camera looking at `target` from `eye` (metres: east, north, up)."""

    def __init__(
        self, eye: Any, target: Any, f: float = 1100.0, cx: float = W / 2, cy: float = H / 2
    ) -> None:
        self.eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - self.eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.vstack([right, down, fwd])
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.f = f

    def project(self, pts: Any) -> tuple[np.ndarray, np.ndarray]:
        p = (np.asarray(pts, float).reshape(-1, 3) - self.eye) @ self.R.T
        z = np.maximum(p[:, 2], 1e-6)
        uv = p[:, :2] / z[:, None] * self.f + [self.K[0, 2], self.K[1, 2]]
        return uv, p[:, 2]

    def pt(self, p: Any) -> tuple[int, int]:
        uv, _ = self.project(p)
        return int(round(uv[0, 0])), int(round(uv[0, 1]))

    def px_per_m(self, p: Any) -> float:
        _, z = self.project(p)
        return self.f / max(float(z[0]), 1e-6)

    def ground(
        self,
        img: np.ndarray,
        tex: np.ndarray,
        tex_to_world: np.ndarray,
        fog: tuple[float, float] | None = None,
        keep_black: bool = True,
    ) -> None:
        """Draw a ground-plane texture (z = 0) onto img; black texels are see-through."""
        m = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], -self.R @ self.eye])
        hmg = m @ tex_to_world
        warped = cv2.warpPerspective(tex, hmg, (W, H), flags=cv2.INTER_LINEAR)
        minv = np.linalg.inv(m)
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        q2 = minv[2, 0] * u + minv[2, 1] * v + minv[2, 2]  # 1 / depth; > 0 in front of the camera
        ok = q2 > 1e-7
        if keep_black:
            ok &= warped.any(axis=2)
        else:  # only where the texture lands
            ones = np.full(tex.shape[:2], 255, np.uint8)
            ok &= cv2.warpPerspective(ones, hmg, (W, H), flags=cv2.INTER_NEAREST) > 0
        if fog is not None:
            depth = 1.0 / np.maximum(q2, 1e-7)
            a = np.clip((fog[1] - depth) / (fog[1] - fog[0]), 0, 1)[..., None]
            warped = (warped * a + img * (1 - a)).astype(np.uint8)
        img[ok] = warped[ok]

    def line(
        self, img: np.ndarray, pts: Any, col: tuple, th: int = 1, closed: bool = False
    ) -> None:
        uv, z = self.project(pts)
        if (z < 1.0).any():
            return
        cv2.polylines(img, [np.round(uv).astype(np.int32)], closed, col, th, cv2.LINE_AA)

    def fill(self, img: np.ndarray, pts: Any, col: tuple, a: float) -> None:
        uv, z = self.project(pts)
        if (z < 1.0).any():
            return
        ov = img.copy()
        cv2.fillPoly(ov, [np.round(uv).astype(np.int32)], col, cv2.LINE_AA)
        cv2.addWeighted(ov, a, img, 1 - a, 0, dst=img)

    def ring(self, img: np.ndarray, centre: Any, r_m: float, col: tuple, th: int = 2) -> None:
        a = np.linspace(0, 2 * np.pi, 48)
        c = np.asarray(centre, float)
        pts = np.column_stack([c[0] + r_m * np.cos(a), c[1] + r_m * np.sin(a), np.full(48, c[2])])
        self.line(img, pts, col, th, closed=True)


def orbit(target: Any, dist: float, az_deg: float, pitch_deg: float, **kw: Any) -> Cam:
    """A camera `dist` metres from target, az_deg clockwise from south, pitch_deg below level."""
    az, pt = np.radians(az_deg), np.radians(pitch_deg)
    t = np.asarray(target, float)
    eye = t + dist * np.array([-np.sin(az) * np.cos(pt), -np.cos(az) * np.cos(pt), np.sin(pt)])
    return Cam(eye, t, **kw)


def encode(frames: Any, dest: Path, crf: int = 23) -> int:
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    assert p.stdin is not None
    for img in frames:
        p.stdin.write(np.ascontiguousarray(img).tobytes())
        n += 1
    p.stdin.close()
    if p.wait():
        raise SystemExit("ffmpeg failed")
    return n


def run(frames: Any, dest: Path, stills: list[float] | None, crf: int = 23) -> None:
    """Encode the video, or with `stills` (seconds) only save those frames as JPEGs."""
    if stills:
        want = {int(round(s * FPS)): s for s in stills}
        for j, img in enumerate(frames):
            if j in want:
                out = dest.with_name(f"{dest.stem}_{want[j]:05.1f}.jpg")
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(out.name, flush=True)
            if j >= max(want):
                break
        return
    n = encode(frames, dest, crf)
    info = {
        "file": dest.name,
        "seconds": round(n / FPS, 2),
        "size_mb": round(dest.stat().st_size / 1e6, 2),
    }
    print(json.dumps(info))
