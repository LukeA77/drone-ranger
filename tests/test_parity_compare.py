import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_onnx as ex  # noqa: E402
from export_onnx import compare  # noqa: E402


def test_identical_detections_have_zero_difference():
    a = (np.array([[0, 0, 10, 10], [50, 50, 5, 5]], float), np.array([0.9, 0.6]), np.array([0, 2]))
    r = compare(a, a)
    assert r["matched"] == 2 and r["max_box_diff_px"] == 0 and r["max_score_diff"] == 0


def test_small_shift_and_class_mismatch():
    a = (np.array([[0, 0, 10, 10], [50, 50, 5, 5]], float), np.array([0.9, 0.6]), np.array([0, 2]))
    b = (
        np.array([[0.5, 0, 10, 10], [50, 50, 5, 5]], float),
        np.array([0.88, 0.6]),
        np.array([0, 3]),
    )
    r = compare(a, b)
    assert r["matched"] == 1
    assert r["max_box_diff_px"] == 0.5
    assert abs(r["max_score_diff"] - 0.02) < 1e-9


def test_rect_height_matches_library_letterbox():
    # 16:9 frames scaled to the long side, then padded up to a multiple of the stride (D-604)
    assert ex.rect_height(640, 756, 1344, 32) == 384
    assert ex.rect_height(960, 756, 1344, 32) == 544
    assert ex.rect_height(960, 1080, 1920, 32) == 544
