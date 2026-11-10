"""Ground image built from camera frames (pipeline/mosaic.py, D-750)."""

import numpy as np
import pandas as pd

from aerialrange.evaluate.auair_world import ground_offset_oblique
from aerialrange.pipeline.mosaic import Camera, build_ground_image, frame_quad, usable_top_row

CAM = Camera(focal_px=400.0, cx=320.0, cy=240.0, width_px=640, height_px=480)


def pose(n=0.0, e=0.0, h=40.0, yaw=0.3, tilt=50.0) -> pd.Series:
    return pd.Series({"n_m": n, "e_m": e, "h_m": h, "yaw_rad": yaw, "tilt_deg": tilt})


def coded_frame(p: pd.Series) -> np.ndarray:
    """A frame whose pixels encode the ground point they see: blue and green = north and east
    (0.5 m per level, offset 128), so the ground image can be checked cell by cell."""
    v, u = np.mgrid[0 : CAM.height_px, 0 : CAM.width_px].astype(float)
    dn, de = ground_offset_oblique(
        u, v, p["h_m"], CAM.focal_px, p["yaw_rad"], p["tilt_deg"], CAM.cx, CAM.cy
    )
    n, e = dn + p["n_m"], de + p["e_m"]
    img = np.zeros((CAM.height_px, CAM.width_px, 3), np.uint8)
    ok = np.isfinite(n) & np.isfinite(e)
    img[..., 0][ok] = np.clip(np.round(n[ok] / 0.5) + 128, 0, 255)
    img[..., 1][ok] = np.clip(np.round(e[ok] / 0.5) + 128, 0, 255)
    img[..., 2] = 255
    return img


def test_usable_rows():
    # looking straight down, every row is far below the horizon
    assert usable_top_row(CAM, 90.0, 10.0) == 0.0
    # the horizon row of a camera tilted 10 degrees is the principal row; 10 degrees below it,
    # the row is f tan(10 deg) further down
    assert np.isclose(usable_top_row(CAM, 10.0, 10.0), CAM.cy)
    assert usable_top_row(CAM, 0.0, 60.0) == CAM.height_px  # nothing that steep is in view


def test_quad_matches_the_ray_model():
    p = pose()
    uv, ne = frame_quad(p, CAM, 10.0)
    dn, de = ground_offset_oblique(uv[:, 0], uv[:, 1], 40.0, 400.0, 0.3, 50.0, 320.0, 240.0)
    assert np.allclose(ne, np.c_[dn, de])


def test_each_cell_shows_the_ground_it_stands_for():
    flight = pd.DataFrame([pose(), pose(n=8.0, e=-5.0, yaw=1.2, tilt=60.0)], index=[1, 2])
    frames = [(f, coded_frame(flight.loc[f])) for f in flight.index]
    gi = build_ground_image(frames, flight, CAM, 10.0, res_m=0.25)
    assert gi is not None and gi.frames_used == 2
    img = gi.image
    filled = img[..., 2] == 255
    assert filled.mean() > 0.3
    rows, cols = np.nonzero(filled)
    n = gi.north_max_m - (rows + 0.5) * gi.res_m
    e = gi.east_min_m + (cols + 0.5) * gi.res_m
    keep = (np.abs(n) < 60) & (np.abs(e) < 60)  # inside the code's range
    got_n = (img[rows, cols, 0].astype(float) - 128) * 0.5
    got_e = (img[rows, cols, 1].astype(float) - 128) * 0.5
    # code rounding (0.25 m) plus interpolation at one cell (0.25 m)
    assert np.median(np.abs(got_n[keep] - n[keep])) < 0.3
    assert np.median(np.abs(got_e[keep] - e[keep])) < 0.3


def test_nothing_usable():
    flight = pd.DataFrame([pose(tilt=0.0)], index=[1])
    assert build_ground_image([(1, coded_frame(flight.loc[1]))], flight, CAM, 60.0) is None
