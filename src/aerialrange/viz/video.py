"""Building blocks for the demo video: text, layout, frame writing and a
size-capped H.264 export.

Frames are BGR uint8 arrays of the video size. Text is drawn with PIL in DejaVu Sans (shipped
with matplotlib, so the charts and the captions share one typeface on any machine). Colours
match the repository figures (`viz.plots`): light surface, near-black text, the status palette
for reliability.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from aerialrange.viz.plots import SURFACE, TEXT, TEXT_2

W, H, FPS = 1280, 720, 25


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


BG, INK, INK_2 = hex_bgr(SURFACE), hex_bgr(TEXT), hex_bgr(TEXT_2)


@cache
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    )
    return ImageFont.truetype(path, size)


def blank(colour: tuple[int, int, int] = BG) -> NDArray[np.uint8]:
    img = np.empty((H, W, 3), np.uint8)
    img[:] = colour
    return img


def text(
    img: NDArray[np.uint8],
    items: Sequence[tuple[str, tuple[int, int], int, tuple[int, int, int], bool]],
    box: tuple[int, int, int] | None = None,
    anchor: str = "la",
) -> NDArray[np.uint8]:
    """Draw several (string, (x, y), size, BGR colour, bold) items in one PIL pass. With `box`
    (a BGR colour), each item gets a filled background rectangle so it reads on footage."""
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    for s, (x, y), size, colour, bold in items:
        f = font(size, bold)
        if box is not None:
            x0, y0, x1, y1 = d.textbbox((x, y), s, font=f, anchor=anchor)
            d.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=box[::-1])
        d.text((x, y), s, font=f, fill=colour[::-1], anchor=anchor)
    img[:] = np.asarray(pil)[:, :, ::-1]
    return img


def wrap(s: str, size: int, width_px: int, bold: bool = False) -> list[str]:
    """Greedy word wrap to a pixel width."""
    f, lines, cur = font(size, bold), [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if f.getlength(trial) <= width_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def fit(img: NDArray[np.uint8], w: int, h: int) -> tuple[NDArray[np.uint8], float]:
    """Resize to fit inside w x h keeping the aspect ratio; returns the image and the scale."""
    import cv2

    s = min(w / img.shape[1], h / img.shape[0])
    size = (round(img.shape[1] * s), round(img.shape[0] * s))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), s


def paste(dst: NDArray[np.uint8], src: NDArray[np.uint8], x: int, y: int) -> None:
    dst[y : y + src.shape[0], x : x + src.shape[1]] = src


def figure_array(fig) -> NDArray[np.uint8]:  # noqa: ANN001 (matplotlib Figure)
    """Render a matplotlib figure to a BGR array."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(rgba[:, :, 2::-1])


def target_kbps(max_mb: float, seconds: float, margin: float = 0.9) -> int:
    """Video bitrate (kbit/s) that keeps a file of `seconds` under `max_mb` megabytes
    (10^6 bytes), leaving `margin` for container overhead and rate-control overshoot."""
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return int(max_mb * 1e6 * 8 * margin / seconds / 1000)


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    return exe


class Writer:
    """Pipe raw frames to ffmpeg as a high-quality master (re-encoded to size by
    `encode_to_size`). Keeps the frame count, and the last frame for fades."""

    def __init__(self, path: Path, size: tuple[int, int] = (W, H), fps: int = FPS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.size, self.fps = size, fps
        self.path, self.n = path, 0
        self.last = np.zeros((size[1], size[0], 3), np.uint8)
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
        cmd += ["-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264"]
        cmd += ["-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img: NDArray[np.uint8], repeat: int = 1) -> None:
        if img.shape != (self.size[1], self.size[0], 3):
            raise ValueError(f"frame shape {img.shape}, expected {self.size[::-1]} x 3")
        b = np.ascontiguousarray(img).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(b)  # type: ignore[union-attr]
        self.n += repeat
        self.last = img

    def hold(self, img: NDArray[np.uint8], seconds: float) -> None:
        self.write(img, max(1, round(seconds * self.fps)))

    def crossfade(self, img: NDArray[np.uint8], seconds: float = 0.4) -> None:
        """Blend from the last frame written into `img`."""
        n = max(1, round(seconds * self.fps))
        a = self.last.astype(np.float32)
        for i in range(1, n + 1):
            t = i / (n + 1)
            self.write((a * (1 - t) + img.astype(np.float32) * t).astype(np.uint8))

    @property
    def seconds(self) -> float:
        return self.n / self.fps

    def close(self) -> None:
        self.proc.stdin.close()  # type: ignore[union-attr]
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed writing the master video")


def encode_to_size(master: Path, out: Path, max_mb: float, seconds: float) -> int:
    """Two-pass H.264 MP4 at the bitrate that fits `max_mb`; no audio. Returns the bitrate."""
    kbps = target_kbps(max_mb, seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".2pass")
    base = [ffmpeg(), "-y", "-loglevel", "error", "-i", str(master), "-c:v", "libx264"]
    base += ["-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-an"]
    base += ["-passlogfile", str(log)]
    subprocess.run(
        [*base, "-pass", "1", "-f", "mp4", "NUL" if _windows() else "/dev/null"], check=True
    )
    subprocess.run([*base, "-pass", "2", "-movflags", "+faststart", str(out)], check=True)
    for p in out.parent.glob(log.name + "*"):
        p.unlink()
    return kbps


def encode_quality(master: Path, out: Path, max_mb: float, seconds: float, crf: int = 21) -> str:
    """H.264 MP4 at constant quality `crf`; if that exceeds `max_mb`, two-pass to fit instead.
    Returns the mode used."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-i", str(master), "-c:v", "libx264"]
    cmd += ["-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an"]
    subprocess.run([*cmd, "-movflags", "+faststart", str(out)], check=True)
    if out.stat().st_size <= max_mb * 1e6:
        return f"crf {crf}"
    return f"two-pass {encode_to_size(master, out, max_mb, seconds)} kbps"


def _windows() -> bool:
    import os

    return os.name == "nt"
