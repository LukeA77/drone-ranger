"""Detection AP on SYNTHETIC detections with hand-checkable answers."""

import numpy as np
import pytest

from aerialrange.detect.evaluate import average_precision, evaluate_detections, match_image

GT = np.array([[0, 0, 10, 10], [50, 50, 10, 20]], float)


def test_perfect_detections_give_ap_one():
    r = evaluate_detections({"img": GT}, {"img": (GT.copy(), np.array([0.9, 0.8]))}, 0.5)
    assert r["ap"] == pytest.approx(1.0)


def test_false_positive_ranked_first_lowers_ap():
    det = np.vstack([[200, 200, 5, 5], GT])
    r = evaluate_detections({"img": GT}, {"img": (det, np.array([0.95, 0.9, 0.8]))}, 0.5)
    assert 0.6 < r["ap"] < 1.0
    assert r["n_gt"] == 2 and r["n_det"] == 3


def test_missed_object_caps_recall():
    r = evaluate_detections({"img": GT}, {"img": (GT[:1], np.array([0.9]))}, 0.5)
    assert r["ap"] == pytest.approx(0.5, abs=0.01)
    assert r["max_recall"] == pytest.approx(0.5)


def test_duplicate_detection_is_false_positive():
    det = np.vstack([GT[0], GT[0], GT[1]])
    tp, _ = match_image(GT, det, np.array([0.9, 0.85, 0.8]), 0.5)
    assert tp.tolist() == [True, False, True]


def test_average_precision_101_point():
    # Precision 1 up to recall 0.5, then 0.5 up to recall 1.
    rec = np.array([0.5, 1.0])
    prec = np.array([1.0, 0.5])
    assert average_precision(rec, prec) == pytest.approx((51 * 1.0 + 50 * 0.5) / 101)


def test_size_bin_restricts_ground_truth():
    gt = {"img": GT}
    det = {"img": (GT.copy(), np.array([0.9, 0.8]))}
    small = evaluate_detections(gt, det, 0.5, height_range=(0, 15))
    assert small["n_gt"] == 1 and small["ap"] == pytest.approx(1.0)


def test_no_ground_truth_gives_nan():
    r = evaluate_detections({"img": np.zeros((0, 4))}, {"img": (GT, np.array([0.9, 0.8]))}, 0.5)
    assert np.isnan(r["ap"])


def test_size_bin_with_an_image_that_has_no_ground_truth():
    """Regression: an image with detections but no ground truth crashed the size-bin path."""
    gt = {"a": GT, "b": np.zeros((0, 4))}
    det = {"a": (GT.copy(), np.array([0.9, 0.8])), "b": (GT[:1], np.array([0.7]))}
    r = evaluate_detections(gt, det, 0.5, height_range=(0, 15))
    assert r["n_gt"] == 1
    assert 0 < r["ap"] <= 1
