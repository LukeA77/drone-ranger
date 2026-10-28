"""AU-AIR loader: names, clip segmentation, units and local NED."""

import json

import numpy as np
import pytest

from aerialrange.data.auair import frames_table, parse_name, segment_clips


def test_parse_name_gives_recording_part_and_index():
    assert parse_name("frame_20190905091750_xx_0000766.jpg") == ("20190905091750", 2, 766)
    assert parse_name("frame_20190829091111_x_0001973.jpg") == ("20190829091111", 1, 1973)
    with pytest.raises(ValueError):
        parse_name("frame_2019_x_1.jpg")


def test_segment_clips_splits_on_long_gaps_within_each_recording():
    rec = np.array(["a", "a", "a", "a", "a", "b", "b"])
    t = np.array([0, 200, 600, 7000, 7200, 0, 200])
    clips = segment_clips(rec, t, gap_ms=6000)
    assert clips.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_segment_clips_orders_by_time_not_input_order():
    rec = np.array(["a", "a", "a"])
    t = np.array([7000, 0, 200])
    assert segment_clips(rec, t, gap_ms=6000).tolist() == [1, 0, 0]


def _frame(name, ms, lat, lon, alt_mm, boxes):
    return {
        "image_name": name,
        "image_width:": 1920.0,
        "image_height": 1080.0,
        "platform": "Parrot Bebop 2",
        "time": {"year": 2019, "month": 9, "day": 5, "hour": 9, "min": 17, "sec": 50, "ms": ms},
        "longtitude": lon,
        "latitude": lat,
        "altitude": alt_mm,
        "linear_x": 0.1,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angle_phi": 0.01,
        "angle_theta": -0.02,
        "angle_psi": 1.5,
        "bbox": boxes,
    }


def test_frames_table_converts_units_and_builds_local_ned(tmp_path):
    box = {"top": 10, "left": 20, "height": 30, "width": 40, "class": 1}
    ann = [
        # second part (_xx) of the recording continues the clock of the first part (_x)
        _frame("frame_20190905091750_xx_0000001.jpg", 600.0, 56.2001, 10.1880, 20000.0, []),
        _frame("frame_20190905091750_x_0000001.jpg", 200.0, 56.2000, 10.1880, 10000.0, [box]),
        _frame("frame_20190905091750_x_0000002.jpg", 400.0, 56.2000, 10.1881, 10500.0, []),
    ]
    (tmp_path / "annotations.json").write_text(json.dumps({"annotations": ann}))
    df = frames_table(tmp_path, gap_ms=6000)
    assert df["image_name"].tolist()[0].endswith("_x_0000001.jpg")  # sorted by time
    np.testing.assert_allclose(df["alt_m"], [10.0, 10.5, 20.0])
    assert df["clip"].nunique() == 1 and df["part"].tolist() == [1, 1, 2]
    # origin is the clip's first frame; 0.0001 deg of latitude is about 11.1 m north
    np.testing.assert_allclose(df[["n_m", "e_m"]].iloc[0], [0.0, 0.0], atol=1e-6)
    assert df["n_m"].iloc[2] == pytest.approx(11.13, abs=0.05)
    assert df["e_m"].iloc[1] == pytest.approx(111320 * np.cos(np.radians(56.2)) * 1e-4, rel=0.01)
    # height above the clip's first fix, positive up (Down = -height)
    # (Earth curvature over 11 m adds about 1e-5 m)
    np.testing.assert_allclose(df["d_m"], [0.0, -0.5, -10.0], atol=1e-3)
    assert json.loads(df["boxes"].iloc[0]) == [box] and df["n_boxes"].tolist() == [1, 0, 0]
    assert str(df["image_path"].iloc[0]).endswith("images/frame_20190905091750_x_0000001.jpg")
