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


def test_counts_by_height_bins_matches_misses_and_switches():
    from aerialrange.track.evaluate import accumulate, counts_by_height

    # object 1 is 20 px tall (bin 16-32), object 2 is 70 px tall (bin 64-128)
    gt = np.array(
        [[f, 1, 10, 10, 10, 20] for f in (1, 2, 3)] + [[f, 2, 100, 100, 30, 70] for f in (1, 2, 3)],
        float,
    )
    # object 1 tracked as id 7 then id 8 (one switch); object 2 matched in frame 1 only
    pred = np.array(
        [
            [1, 7, 10, 10, 10, 20],
            [2, 7, 10, 10, 10, 20],
            [3, 8, 10, 10, 10, 20],
            [1, 9, 100, 100, 30, 70],
        ],
        float,
    )
    c = counts_by_height(accumulate(gt, pred, {}, 0.5), gt)
    np.testing.assert_array_equal(c[:, 1], [3, 3, 1])  # 16-32: total, matched, switches
    np.testing.assert_array_equal(c[:, 3], [3, 1, 0])  # 64-128
    assert c[0, [0, 2, 4]].sum() == 0
