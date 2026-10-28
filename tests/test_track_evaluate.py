"""Tracking evaluation on SYNTHETIC tracks with hand-computed metrics."""

import numpy as np
import pytest

mm = pytest.importorskip("motmetrics")

from aerialrange.track.evaluate import (  # noqa: E402
    accumulate,
    iou_distance,
    read_ignored,
    read_rows,
    summarise,
)

GT = np.array(
    [
        [1, 1, 0, 0, 10, 10],
        [2, 1, 1, 0, 10, 10],
        [3, 1, 2, 0, 10, 10],
        [1, 2, 50, 50, 10, 10],
        [2, 2, 50, 50, 10, 10],
        [3, 2, 50, 50, 10, 10],
    ],
    float,
)


def test_perfect_tracks():
    s = summarise({"seq": accumulate(GT, GT.copy(), {}, 0.5)})
    assert s["OVERALL"]["mota"] == pytest.approx(1.0)
    assert s["OVERALL"]["idf1"] == pytest.approx(1.0)
    assert s["OVERALL"]["num_switches"] == 0


def test_identity_switch_counted():
    pred = GT.copy()
    pred[2, 1] = 7  # track 1 changes identity at frame 3
    s = summarise({"seq": accumulate(GT, pred, {}, 0.5)})["OVERALL"]
    assert s["num_switches"] == 1
    assert s["mota"] == pytest.approx(1 - 1 / 6)


def test_low_iou_is_a_miss_and_false_positive():
    pred = GT.copy()
    pred[0, 2:4] = [6, 6]  # IoU = 16 / 184 < 0.5
    s = summarise({"seq": accumulate(GT, pred, {}, 0.5)})["OVERALL"]
    assert s["num_misses"] == 1 and s["num_false_positives"] == 1


def test_predictions_in_ignored_regions_are_removed():
    pred = np.vstack([GT, [[2, 9, 200, 200, 10, 10]]])
    ign = {2: np.array([[190.0, 190.0, 40.0, 40.0]])}
    with_ign = summarise({"s": accumulate(GT, pred, ign, 0.5)})["OVERALL"]
    without = summarise({"s": accumulate(GT, pred, {}, 0.5)})["OVERALL"]
    assert with_ign["num_false_positives"] == 0
    assert without["num_false_positives"] == 1


def test_readers(tmp_path):
    (tmp_path / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n")
    (tmp_path / "ign.txt").write_text("2,0,0,5,5\n2,10,10,5,5\n")
    (tmp_path / "empty.txt").write_text("\n")
    assert read_rows(tmp_path / "gt.txt", 6).shape == (1, 6)
    assert read_rows(tmp_path / "empty.txt", 6).shape == (0, 6)
    assert read_ignored(tmp_path / "ign.txt")[2].shape == (2, 4)


def test_iou_distance_values_and_threshold():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [100, 100, 5, 5]], float)
    d = iou_distance(a, b, 0.3)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(1 - 50 / 150)  # IoU 1/3 >= 0.3
    assert np.isnan(d[0, 2])
    assert np.isnan(iou_distance(a, b[1:2], 0.5)[0, 0])  # IoU 1/3 < 0.5
    assert iou_distance(a, np.zeros((0, 4)), 0.5).shape == (1, 0)
