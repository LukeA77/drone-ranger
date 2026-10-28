"""Tracking with a custom detection function (used for tiled detection, B3)."""

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from aerialrange.track.runner import DetSet, run_sequence_custom  # noqa: E402
from aerialrange.utils.config import REPO_ROOT  # noqa: E402


def test_detset_boolean_indexing():
    d = DetSet(np.zeros((3, 4)), np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2]))
    sub = d[d.conf > 0.3]
    assert len(sub) == 2 and sub.cls.tolist() == [1, 2]


@pytest.mark.parametrize("cfg", ["bytetrack.yaml", "botsort_cmc.yaml"])
def test_steadily_moving_box_keeps_one_identity(cfg):
    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(12)]

    def detect(img_index):
        x = 40 + 6 * img_index
        return (np.array([[x, 100.0, 20.0, 30.0]]), np.array([0.9]), np.array([3]))

    calls = iter(range(len(frames)))
    rows, timing = run_sequence_custom(
        lambda img: detect(next(calls)),
        frames,
        str(REPO_ROOT / "configs" / "trackers" / cfg),
        {3: "vehicle"},
    )
    v = rows["vehicle"]
    assert len(v) >= 10  # tracks are confirmed after the first frames
    assert len(np.unique(v[:, 1])) == 1
    np.testing.assert_allclose(v[-1, 2:6], [40 + 6 * 11, 100, 20, 30], atol=2.0)
    assert timing["frames"] == 12


def test_detset_exposes_corner_boxes_for_camera_motion_compensation():
    """Regression: BoT-SORT's GMC masks detections using `xyxy`; without it GMC was silently
    disabled for tiled tracking."""
    d = DetSet(np.array([[50.0, 40.0, 20.0, 10.0]]), np.array([0.9]), np.array([3.0]))
    np.testing.assert_allclose(d.xyxy, [[40.0, 35.0, 60.0, 45.0]])
    assert d[np.array([True])].xyxy.shape == (1, 4)


@pytest.mark.parametrize("cfg", ["bytetrack.yaml", "botsort_cmc.yaml"])
def test_detection_stride_outputs_predicted_boxes_on_skipped_frames(cfg):
    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(16)]
    calls = []

    def detect(img):
        k = len(calls)
        calls.append(k)
        return (np.array([[40.0 + 6 * k * 2, 100, 20, 30]]), np.array([0.9]), np.array([3]))

    rows, _ = run_sequence_custom(
        detect, frames, str(REPO_ROOT / "configs" / "trackers" / cfg), {3: "vehicle"}, det_stride=2
    )
    v = rows["vehicle"]
    assert len(calls) == 8  # detector ran on every 2nd frame only
    assert len(np.unique(v[:, 1])) == 1  # one identity throughout
    later = v[v[:, 0] >= 6]
    assert set(later[:, 0].astype(int)) == set(range(6, 17))  # a box on every frame, skipped too
    # predicted boxes on skipped frames sit between their detected neighbours
    x = dict(zip(later[:, 0].astype(int), later[:, 2], strict=True))
    assert x[8] < x[10] < x[12]
