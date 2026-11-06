"""SynDrone loader on a small synthetic fixture (labelled synthetic; no dataset files)."""

import json

import cv2
import numpy as np
import pytest

from aerialrange.data.syndrone import (
    BOTTOM,
    SynDroneSequence,
    decode_depth,
    decode_semantic,
    encode_depth,
    parse_boxes,
    parse_camera,
)
from aerialrange.data.syndrone_archive import MemberStore, member_name


def _corners(c, e):
    """SynDrone corner order for an axis-aligned box (get_corners_from_bb)."""
    s = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]
    return [[c[i] + si * e[i] for i, si in enumerate(sg)] for sg in s]


def test_member_names():
    assert member_name("Town01", "rgb", 7, 20) == "Town01_Opt_120/ClearNoon/height20m/rgb/00007.jpg"
    assert member_name("Town07", "bboxes", 2999) == "Town07_Opt_120/ClearNoon/bboxes/02999.json"
    with pytest.raises(ValueError):
        member_name("Town01", "depth", 0, 30)


def test_depth_round_trip_quantisation():
    d = np.array([[0.0, 1.0, 80.123], [500.0, 999.9, 1000.0]], np.float32)
    back = decode_depth(encode_depth(d))
    assert np.all(np.abs(back - d) <= 1000.0 / 65535.0 + 1e-6)
    assert np.all(back <= d + 1e-4)  # floor quantisation never reads long


def test_semantic_grey_and_colour():
    lab = np.array([[40, 100], [7, 0]], np.uint8)
    ok, g = cv2.imencode(".png", lab)
    assert np.array_equal(decode_semantic(g.tobytes()), lab)
    col = np.zeros((2, 2, 3), np.uint8)
    col[..., 2] = lab  # red channel in OpenCV's BGR order
    ok, c = cv2.imencode(".png", col)
    assert np.array_equal(decode_semantic(c.tobytes()), lab)


def test_boxes_level_prefix_and_footprint():
    items = [
        {"corners": _corners((0, 0, 1), (2, 1, 1)), "class": [100], "id": 0},
        {"corners": _corners((5, 0, 1), (2, 1, 1)), "class": [100], "id": 1},
        {"corners": _corners((9, 9, 0.9), (0.3, 0.3, 0.9)), "class": [40], "id": 4127},
        {"corners": _corners((3, 3, 1), (2, 1, 1)), "class": [101, 41], "id": 2},
    ]
    b = parse_boxes(json.dumps(items))
    assert b.level.tolist() == [True, True, False, False]  # id 2 after an actor is not level
    assert b.tags.tolist() == [100, 100, 40, 101]
    assert np.allclose(b.footprint_centre[2], [9, 9, 0.0])
    assert np.allclose(b.corners[0, BOTTOM, 2], 0.0)
    assert len(b.select(~b.level)) == 2


def test_camera_record_pose():
    r = parse_camera('{"x": 1.0, "y": 2.0, "z": 80.0, "pitch": -90.0, "yaw": 3.4, "roll": 0}')
    assert r.h == 80.0
    assert np.allclose(r.pose(z0=2.5).p_w, [1.0, 2.0, -82.5])


def test_sequence_reads_local_cache(tmp_path):
    store = MemberStore(tmp_path)
    seq = SynDroneSequence(store, "Town01", 50)
    for k in range(3):
        p = store.local_path(seq.member("camera", k))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"x": k, "y": 0, "z": 50.0, "pitch": -60.0, "yaw": 0, "roll": 0}))
    p = store.local_path(seq.member("depth", 1))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(encode_depth(np.full((4, 6), 57.7)))
    recs = seq.camera_records(range(3))
    assert [r.x for r in recs] == [0.0, 1.0, 2.0]
    assert seq.depth(1).shape == (4, 6)
    with pytest.raises(FileNotFoundError):  # not cached and no remote configured
        seq.depth(2)


def test_store_quarantines_a_cached_member_that_fails_its_crc(tmp_path):
    """A zero-filled cached file (sleep mid-write) is never returned as data (2026-11-05)."""
    import json
    import zlib

    name = member_name("Town01", "camera", 7, 20)
    good = b'{"x": 1, "y": 2, "z": 20, "pitch": -30, "yaw": 0, "roll": 0}'
    (tmp_path / "_index").mkdir()
    (tmp_path / "_index" / "Town01_Opt_120_color.zip.index.json").write_text(
        json.dumps({"members": [[name, 0, len(good), len(good), zlib.crc32(good), 8]]})
    )
    (tmp_path / name).parent.mkdir(parents=True)
    (tmp_path / name).write_bytes(bytes(len(good)))
    store = MemberStore(tmp_path)  # no base_url: cannot refetch, so must fail loudly
    with pytest.warns(UserWarning, match="quarantined"), pytest.raises(FileNotFoundError):
        store.read(name)
    assert (tmp_path / "_quarantine" / name).read_bytes() == bytes(len(good))
    assert not (tmp_path / name).exists()

    (tmp_path / name).write_bytes(good)  # a correct copy is used as is
    assert store.read(name) == good


def test_parse_boxes_drops_exact_duplicate_actor_and_rejects_conflicts():
    """Town10HD lists one actor twice with identical content in every box file (D-709)."""
    import json

    from aerialrange.data.syndrone import parse_boxes

    def box(i, cls, dz=0.0):
        return {"class": cls, "id": i, "corners": [[0, 0, dz + z] for z in range(8)]}

    level = [box(0, [1]), box(1, [2])]  # map boxes: id == position
    bike = box(122, [104, 41])
    b = parse_boxes(json.dumps([*level, box(40, [100]), bike, dict(bike)]))
    assert b.ids.tolist() == [0, 1, 40, 122]
    assert b.level.tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="different boxes"):
        parse_boxes(json.dumps([*level, bike, box(122, [104, 41], dz=1.0)]))
