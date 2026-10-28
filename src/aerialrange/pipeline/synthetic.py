"""A small synthetic flight for the one-command demo and the end-to-end test (no datasets).

A camera at a known height and tilt flies north over textured flat ground; rectangular
"vehicles" (flat footprints on the ground, in neutral greys so they never read as the overlay's
status colours) are parked or drive at known speeds. Each frame's
image is rendered by projecting the ground into the camera (`project_ground`), so geometry is
exact by construction. Writes frames/, boxes.csv (per-frame axis-aligned boxes of the projected
footprints, class 3 = car in the detector's VisDrone ids), metadata.csv (camera pose per frame)
and truth.csv (true footprint centres). Synthetic data, labelled as such.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aerialrange.evaluate.auair_world import project_ground

W, H, F = 960, 540, 630.0  # half-size frames; focal scaled with them


def make_flight(out: Path, n_frames: int = 60, fps: float = 5.0, seed: int = 0) -> dict:
    out = Path(out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # tileable ground texture: noise blurred by circular (FFT) convolution, so it wraps cleanly
    fy, fx = np.meshgrid(np.fft.fftfreq(800), np.fft.fftfreq(800), indexing="ij")
    blur = np.exp(-2 * (np.pi * 3) ** 2 * (fx**2 + fy**2))
    tex = np.real(np.fft.ifft2(np.fft.fft2(rng.random((800, 800))) * blur)).astype(np.float32)
    tex = cv2.cvtColor(cv2.normalize(tex, None, 60, 190, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    h, tilt, yaw, v_cam = 25.0, 50.0, 0.0, 2.0
    # vehicles: (north0, east0, v_north, v_east, length, width, colour)
    cars = [
        (40.0, -6.0, 0.0, 0.0, 4.5, 1.9, (235, 235, 235)),  # parked, white
        (25.0, 4.0, 6.0, 0.0, 4.4, 1.8, (35, 35, 35)),  # 6 m/s north, black
        (55.0, 10.0, 0.0, -2.5, 4.8, 2.0, (150, 140, 130)),  # 2.5 m/s west, grey
    ]
    boxes, meta, truth = [], [], []
    vv, uu = np.mgrid[0:H, 0:W]
    for k in range(n_frames):
        t = k / fps
        cam_n = v_cam * t
        # ground under each pixel (rays from this camera), textured by world position
        x = (uu - W / 2) / F
        y = (vv - H / 2) / F
        c, s = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
        fwd, down = c - s * y, s + c * y
        with np.errstate(divide="ignore", invalid="ignore"):
            kk = np.where(down > 1e-6, h / down, np.nan)
        gn, ge = cam_n + kk * fwd, kk * x
        sky = ~np.isfinite(gn)
        ti = np.floor(np.nan_to_num(gn) * 8).astype(int) % 800  # floor: no seam at 0
        tj = np.floor(np.nan_to_num(ge) * 8).astype(int) % 800
        img = tex[ti, tj].copy()
        img[sky] = (210, 190, 160)
        for cid, (n0, e0, vn, ve, L, Wd, col) in enumerate(cars, start=1):
            cn, ce = n0 + vn * t, e0 + ve * t
            corners = np.array(
                [[cn + a * L / 2, ce + b * Wd / 2] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            )
            u, v = project_ground(
                corners[:, 0] - cam_n, corners[:, 1], h, F, yaw, tilt, W / 2, H / 2
            )
            if not np.isfinite(u).all():
                continue
            poly = np.column_stack([u, v]).astype(np.int32)
            cv2.fillPoly(img, [poly], col)
            x0, y0, x1, y1 = u.min(), v.min(), u.max(), v.max()
            if x1 < 0 or y1 < 0 or x0 > W or y0 > H:
                continue
            boxes.append((k + 1, x0, y0, x1 - x0, y1 - y0, 0.9, 3))
            truth.append((k + 1, cid, cn, ce))
        cv2.imwrite(str(out / "frames" / f"{k + 1:06d}.jpg"), img)
        meta.append((k + 1, t, cam_n, 0.0, h, yaw, tilt))
    pd.DataFrame(boxes, columns=["frame", "left", "top", "width", "height", "score", "cls"]).to_csv(
        out / "boxes.csv", index=False
    )
    pd.DataFrame(meta, columns=["frame", "t_s", "n_m", "e_m", "h_m", "yaw_rad", "tilt_deg"]).to_csv(
        out / "metadata.csv", index=False
    )
    pd.DataFrame(truth, columns=["frame", "car", "n_m", "e_m"]).to_csv(
        out / "truth.csv", index=False
    )
    return {
        "frames": n_frames,
        "focal_px": F,
        "cx": W / 2,
        "cy": H / 2,
        "speeds_ms": [0.0, 6.0, 2.5],
    }
