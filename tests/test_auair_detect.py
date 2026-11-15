"""AU-AIR scoring for the version 2 detector uses version 1's frames and class mapping."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import auair_detect as ad  # noqa: E402


def test_truth_maps_classes_like_version_1() -> None:
    boxes = [
        {"class": 0, "left": 1, "top": 2, "width": 3, "height": 4},  # Human
        {"class": 1, "left": 5, "top": 6, "width": 7, "height": 8},  # Car
        {"class": 6, "left": 9, "top": 9, "width": 9, "height": 9},  # Bus
        {"class": 4, "left": 0, "top": 0, "width": 1, "height": 1},  # Motorbike: not scored
        {"class": 7, "left": 0, "top": 0, "width": 1, "height": 1},  # Trailer: not scored
    ]
    np.testing.assert_array_equal(ad.truth(boxes, "person"), [[1, 2, 3, 4]])
    assert ad.truth(boxes, "vehicle").shape == (2, 4)
    assert ad.truth([], "person").shape == (0, 4)


def test_stride_frames_takes_every_nth_in_time_order() -> None:
    frames = pd.DataFrame(
        {"clip": [3] * 25 + [4] * 5, "t_ms": list(range(24, -1, -1)) + list(range(5))}
    )
    sel = ad.stride_frames(frames, 3, 10)
    assert list(sel["t_ms"]) == [0, 10, 20]
    assert len(ad.stride_frames(frames, 4, 10)) == 1
