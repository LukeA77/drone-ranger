"""SynDrone truth on synthetic scenes with known answers."""

import json

import numpy as np
import pytest

from aerialrange.data.syndrone import CameraRecord, parse_boxes
from aerialrange.evaluate.syndrone_truth import (
    View,
    _nearest_on_polygon,
    box_frames,
    estimate_z0,
    huber_location,
    object_truth,
)
from syndrone_synth import box_corners, render

G = 3.0  # synthetic ground height (Unreal z)
CAR = (2.2, 1.0, 0.75)  # half sizes


def _boxes(items):
    return parse_boxes(
        json.dumps([{"corners": c.tolist(), "class": [t], "id": i} for i, (c, t) in items])
    )


def _scene(rec, z0, items, occluders=()):
    objs = [(c, t) for _, (c, t) in items] + list(occluders)
    cam, depth, sem = render(rec, z0, G, objs)
    return View(int(rec.h), rec, depth, sem, cam)


def _cars(n=4):
    out = []
    for k in range(n):
        c = box_corners((30.0 + 7 * k, -6.0 + 4 * k, G + CAR[2]), CAR, yaw_deg=20 * k)
        out.append((1000 + k, (c, 100)))
    return out


@pytest.mark.parametrize("h,pitch", [(20, -30.0), (50, -60.0), (80, -90.0)])
def test_z0_recovered_from_road_around_cars(h, pitch):
    true_z0 = G + 0.4  # base height need not equal the ground
    x = 40.0 if pitch == -90 else 40.0 - h / np.tan(np.radians(-pitch))
    rec = CameraRecord(x, 0.0, float(h), pitch, 10.0, 0.0)
    items = _cars()
    view = _scene(rec, true_z0, items)
    fit = estimate_z0([view], _boxes(items))
    assert fit.n_boxes >= 2
    # Vehicles stand on the road at G; the camera is at z0 + h. The offset recovered is
    # (box bottom) - (road z relative to z0 = 0) = z0.
    assert fit.z0 == pytest.approx(true_z0, abs=0.02)


def test_object_truth_unoccluded_and_occluded():
    rec = CameraRecord(40.0, 0.0, 80.0, -90.0, 0.0, 0.0)
    car = box_corners((40.0, 0.0, G + CAR[2]), CAR)
    items = [(1000, (car, 100))]
    view = _scene(rec, G, items)
    (row,) = object_truth(view, _boxes(items), z0=G)
    assert row["visibility"] == pytest.approx(1.0)
    assert row["range_m"] == pytest.approx(80.0, abs=1e-6)  # straight down to the footprint
    assert row["depression_deg"] == pytest.approx(90.0)
    assert np.allclose(row["foot_ue"], [40.0, 0.0, G])
    # A tall wall (tag 11) over the car's x > 40 half hides about half of it.
    wall = box_corners((41.25, 0.0, G + 10.0), (1.25, 3.0, 2.0))
    view2 = _scene(rec, G, items, occluders=[(wall, 11)])
    (row2,) = object_truth(view2, _boxes(items), z0=G)
    assert 0.35 < row2["visibility"] < 0.65


def test_projected_and_mask_boxes_agree_for_a_box_shaped_object():
    rec = CameraRecord(10.0, 0.0, 50.0, -60.0, 0.0, 0.0)
    car = box_corners((38.0, 2.0, G + CAR[2]), CAR, yaw_deg=30)
    items = [(7, (car, 100))]
    view = _scene(rec, G, items)
    (row,) = object_truth(view, _boxes(items), z0=G)
    assert np.allclose(row["box3d_proj"], row["box_mask"], atol=1.0)


def test_wrong_z0_moves_projection():
    rec = CameraRecord(10.0, 0.0, 20.0, -30.0, 0.0, 0.0)
    car = box_corners((40.0, 0.0, G + CAR[2]), CAR)
    items = [(7, (car, 100))]
    view = _scene(rec, G, items)
    (good,) = object_truth(view, _boxes(items), z0=G)
    (bad,) = object_truth(view, _boxes(items), z0=G + 2.0)
    err = lambda r: np.abs(np.subtract(r["box3d_proj"], r["box_mask"])).max()  # noqa: E731
    assert err(good) <= 1.0 < err(bad)


def test_near_edge_and_level_flag():
    rec = CameraRecord(0.0, 0.0, 20.0, -30.0, 0.0, 0.0)
    car = box_corners((30.0, 0.0, G + CAR[2]), CAR)
    items = [(0, (car, 100))]  # id 0 in first position: a level box
    view = _scene(rec, G, items)
    (row,) = object_truth(view, _boxes(items), z0=G)
    assert row["level"] is True
    assert row["range_near_m"] < row["range_m"]
    near = np.hypot(30.0 - CAR[0], 20.0 + 0.0)  # z0 = G so the camera is 20 m above ground
    assert row["range_near_m"] == pytest.approx(near, abs=1e-6)


def test_box_frames_from_corners():
    c = box_corners((1.0, 2.0, 3.0), (2.0, 1.0, 0.5), yaw_deg=90)
    cen, axes, half = box_frames(c[None])
    assert np.allclose(cen[0], [1, 2, 3]) and np.allclose(half[0], [2, 1, 0.5])
    assert np.allclose(axes[0][0], [0, 1, 0], atol=1e-12)


def test_huber_resists_outliers():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(1.0, 0.05, 200), [50.0, -40.0, 30.0]])
    mu, se, s = huber_location(x, 1.345)
    assert mu == pytest.approx(1.0, abs=0.02) and 0 < se < 0.01


def test_nearest_on_polygon():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert np.allclose(_nearest_on_polygon(sq, np.array([5.0, 1.0])), [2, 1])
    assert np.allclose(_nearest_on_polygon(sq, np.array([1.0, 1.0])), [1, 1])
