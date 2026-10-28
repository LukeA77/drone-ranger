"""Frame overlay for the annotated video.

Boxes are coloured by reliability with the reference status palette (good / warning /
critical), and colour never stands alone: every box carries a text label (range with its
sigma and relative bearing, or the reason there is no range) and the legend names the three
states. Faint ground-distance rings are drawn around the point below the drone, with a frame
timer and a dataset credit line. Overlay time is excluded from the latency budget (10.1).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from aerialrange.evaluate.auair_world import project_ground
from aerialrange.pipeline.ranging import CameraPose

# Status palette (dataviz reference instance), as BGR for OpenCV.
STATUS = {
    "reliable": (0x0C, 0xA3, 0x0C),  # good  #0ca30c
    "uncertain": (0x19, 0xB2, 0xFA),  # warning #fab219
    "invalid": (0x3B, 0x3B, 0xD0),  # critical #d03b3b
}
INK, HALO, RING = (250, 250, 250), (20, 20, 20), (235, 235, 235)


def _text(img: NDArray, s: str, org: tuple[int, int], scale: float, colour: tuple = INK) -> None:
    """Text on a dark label background (readable on any ground texture)."""
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), HALO, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_rings(
    img: NDArray, pose: CameraPose, focal_px: float, cx: float, cy: float, rings_m: list
) -> None:
    """Faint circles on the ground at the given horizontal distances from the point below the
    drone, projected into the image."""
    if not np.isfinite(pose.tilt_deg):
        return
    overlay = img.copy()
    a = np.linspace(0, 2 * np.pi, 180)
    for r in rings_m:
        u, v = project_ground(
            r * np.cos(a), r * np.sin(a), pose.h_m, focal_px, pose.yaw_rad, pose.tilt_deg, cx, cy
        )
        ok = np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 1e5) & (np.abs(v) < 1e5)
        pts = np.column_stack([u, v])
        for seg in np.split(pts, np.flatnonzero(~ok)):
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) > 1:
                cv2.polylines(overlay, [seg.astype(np.int32)], False, RING, 1, cv2.LINE_AA)
        j = np.flatnonzero(ok & (u > 0) & (u < img.shape[1]) & (v > 0) & (v < img.shape[0]))
        if len(j):
            k = j[np.argmax(v[j])]
            _text(overlay, f"{r:g} m", (int(u[k]) + 4, int(v[k]) - 4), 0.45, RING)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)


def draw_frame(
    img: NDArray,
    frame: int,
    t_s: float,
    rows: list[dict],
    pose: CameraPose,
    focal_px: float,
    cx: float,
    cy: float,
    rings_m: list,
    scale: float,
    credit: str,
) -> NDArray:
    """Annotate one frame in place and return it. `rows` hold box, flag, range, sigma and
    relative azimuth per track."""
    draw_rings(img, pose, focal_px, cx, cy, rings_m)
    for r in rows:
        x, y, w, h = (int(round(v)) for v in (r["left"], r["top"], r["width"], r["height"]))
        col = STATUS[r["flag"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2 if r["flag"] != "invalid" else 1)
        if r["flag"] == "invalid":
            label = f"#{r['track_id']} no range ({r['reason'].replace('_', ' ')})"
        else:
            label = (
                f"#{r['track_id']} {r['range_m']:.0f} ± {r['range_sigma_m']:.0f} m "
                f"{r['relative_azimuth_deg']:+.0f}°"
            )
        _text(img, label.replace("°", " deg").replace("±", "+/-"), (x, max(12, y - 5)), scale * 0.8)
    # legend: colour plus words, top-left
    y0 = 30
    for name, col in STATUS.items():
        cv2.rectangle(img, (14, y0 - 12), (30, y0 + 2), col, -1)
        _text(img, name, (38, y0), scale)
        y0 += 24
    _text(img, f"frame {frame}  t {t_s:6.1f} s", (14, y0 + 6), scale)
    _text(img, credit, (14, img.shape[0] - 14), scale * 0.75)
    return img
