"""Linking AU-AIR annotated boxes into tracks through the camera motion (D-513)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import auair_stationary as st  # noqa: E402


def _shift(dx):
    return np.array([[1.0, 0, dx], [0, 1.0, 0], [0, 0, 1.0]])


def test_parked_box_keeps_its_id_under_a_large_camera_pan():
    # the camera pans 60 px/frame: without compensation a 50 px box would not overlap itself
    frames = [np.array([[100.0 - 60 * k, 50, 50, 30]]) for k in range(4)]
    frames[2] = np.vstack([frames[2], [[400.0, 200, 40, 40]]])  # a new object appears
    tr = st.link_truth(frames, [_shift(-60)] * 3)
    ids_first = tr[tr[:, 2] < 150][:, 1]
    assert set(ids_first) == {1}
    assert tr[(tr[:, 0] == 3) & (tr[:, 2] == 400.0)][0, 1] == 2


def test_warp_boxes_and_iou():
    b = np.array([[10.0, 10, 20, 10]])
    np.testing.assert_allclose(st.warp_boxes(b, _shift(5)), [[15.0, 10, 20, 10]])
    assert st.iou_matrix(b, b)[0, 0] == 1.0
    assert st.iou_matrix(b, np.array([[100.0, 100, 5, 5]]))[0, 0] == 0.0


def test_link_bridges_frames_where_the_annotation_is_missing():
    frames = [np.array([[100.0 - 60 * k, 50, 50, 30]]) for k in range(6)]
    frames[2] = np.zeros((0, 4))
    frames[3] = np.zeros((0, 4))  # missing for two frames
    tr = st.link_truth(frames, [_shift(-60)] * 5)
    assert set(tr[:, 1]) == {1}
    assert set(st.link_truth(frames, [_shift(-60)] * 5, max_gap=1)[:, 1]) == {1, 2}
