"""Tracking with a custom detection function, for the project's tracker and, where
version 1's library is installed, its library trackers (scripts/ultralytics_v1.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from aerialrange.track.runner import DetSet, make_tracker, run_sequence_custom
from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ultralytics_v1  # noqa: E402

LIBRARY = ("bytetrack.yaml", "botsort_cmc.yaml")
TRACKERS = ["motion.yaml", *LIBRARY]


def _cfg(name):
    if name in LIBRARY:
        pytest.importorskip("ultralytics")
    return str(REPO_ROOT / "configs" / "trackers" / name)


def test_detset_boolean_indexing():
    d = DetSet(np.zeros((3, 4)), np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2]))
    sub = d[d.conf > 0.3]
    assert len(sub) == 2 and sub.cls.tolist() == [1, 2]


@pytest.mark.parametrize("cfg", TRACKERS)
def test_steadily_moving_box_keeps_one_identity(cfg):
    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(12)]

    def detect(img_index):
        x = 40 + 6 * img_index
        return (np.array([[x, 100.0, 20.0, 30.0]]), np.array([0.9]), np.array([3]))

    calls = iter(range(len(frames)))
    rows, timing = run_sequence_custom(
        lambda img: detect(next(calls)),
        frames,
        _cfg(cfg),
        {3: "vehicle"},
        build_tracker=ultralytics_v1.make_tracker,
    )
    v = rows["vehicle"]
    assert len(v) >= 10  # tracks are confirmed after the first frames
    assert len(np.unique(v[:, 1])) == 1
    np.testing.assert_allclose(v[-1, 2:6], [40 + 6 * 11, 100, 20, 30], atol=2.0)
    assert timing["frames"] == 12


def test_detset_exposes_corner_boxes_for_camera_motion_compensation():
    """Regression: version 1's BoT-SORT masked detections using `xyxy`; without it camera-
    motion compensation was silently disabled for tiled tracking."""
    d = DetSet(np.array([[50.0, 40.0, 20.0, 10.0]]), np.array([0.9]), np.array([3.0]))
    np.testing.assert_allclose(d.xyxy, [[40.0, 35.0, 60.0, 45.0]])
    assert d[np.array([True])].xyxy.shape == (1, 4)


@pytest.mark.parametrize("cfg", TRACKERS)
def test_detection_stride_outputs_predicted_boxes_on_skipped_frames(cfg):
    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(16)]
    calls = []

    def detect(img):
        k = len(calls)
        calls.append(k)
        return (np.array([[40.0 + 6 * k * 2, 100, 20, 30]]), np.array([0.9]), np.array([3]))

    rows, _ = run_sequence_custom(
        detect,
        frames,
        _cfg(cfg),
        {3: "vehicle"},
        det_stride=2,
        build_tracker=ultralytics_v1.make_tracker,
    )
    v = rows["vehicle"]
    assert len(calls) == 8  # detector ran on every 2nd frame only
    assert len(np.unique(v[:, 1])) == 1  # one identity throughout
    later = v[v[:, 0] >= 6]
    assert set(later[:, 0].astype(int)) == set(range(6, 17))  # a box on every frame, skipped too
    # predicted boxes on skipped frames sit between their detected neighbours
    x = dict(zip(later[:, 0].astype(int), later[:, 2], strict=True))
    assert x[8] < x[10] < x[12]


def test_package_tracker_refuses_library_tracker_configs():
    """The package builds only its own tracker; version 1's are in scripts/ultralytics_v1.py."""
    with pytest.raises(ValueError, match="ultralytics_v1"):
        make_tracker(str(REPO_ROOT / "configs" / "trackers" / "bytetrack.yaml"))


def test_version1_helper_builds_the_project_tracker_without_the_library():
    from aerialrange.track.tracker import MotionTracker

    t = ultralytics_v1.make_tracker(str(REPO_ROOT / "configs" / "trackers" / "motion.yaml"))
    assert isinstance(t, MotionTracker)
