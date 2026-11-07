"""RF-DETR and D-FINE ONNX post-processing and the detector factory (no model file needed)."""

from __future__ import annotations

import numpy as np
import pytest

from aerialrange.detect.factory import make_detector
from aerialrange.detect.onnx_detectors import DetrOnnx, DFINEOnnx, RFDETROnnx


def _bare(
    score_floor: float = 0.1, max_det: int = 300, cls: type[DetrOnnx] = DFINEOnnx
) -> DetrOnnx:
    det = cls.__new__(cls)
    det.h = det.w = 640
    det.score_floor = score_floor
    det.max_det = max_det
    return det


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_rfdetr_slots_are_zero_based():
    # RF-DETR renumbers categories from 0 in training: slot 0 is pedestrian, slot 10 unused.
    det = _bare(cls=RFDETROnnx)
    boxes = np.tile(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), (3, 1))
    logits = np.full((3, 11), -20.0, np.float32)
    logits[0, 0] = _logit(0.9)  # pedestrian = class index 0
    logits[1, 3] = _logit(0.8)  # car = class index 3
    logits[2, 10] = _logit(0.95)  # outside the ten classes: dropped
    d = det.postprocess(boxes, logits, 100, 100)
    assert d.cls.tolist() == [0, 3]


def test_dfine_slots_are_category_ids():
    det = _bare()
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.75, 0.1, 0.1]], np.float32)
    logits = np.full((2, 11), -20.0, np.float32)
    logits[0, 4] = _logit(0.9)  # slot 4 = VisDrone category 4 (car) = class index 3
    logits[1, 1] = _logit(0.6)  # slot 1 = pedestrian = class index 0
    logits[1, 0] = _logit(0.99)  # slot 0 is unused and must be dropped
    d = det.postprocess(boxes, logits, img_h=1000, img_w=2000)
    assert d.cls.tolist() == [3, 0]
    np.testing.assert_allclose(d.score, [0.9, 0.6], rtol=1e-6)
    np.testing.assert_allclose(d.xyxy[0], [800, 300, 1200, 700], atol=1e-3)
    np.testing.assert_allclose(d.xywh[1], [500, 750, 200, 100], atol=1e-3)


def test_postprocess_keeps_top_pairs_over_queries_and_classes():
    det = _bare(score_floor=0.0, max_det=3)
    boxes = np.tile(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), (2, 1))
    logits = np.full((2, 4), -20.0, np.float32)
    # One query may contribute several classes, as in the library's top-k over all pairs.
    logits[0, 1], logits[0, 2], logits[1, 3], logits[1, 1] = (
        _logit(p) for p in (0.8, 0.7, 0.6, 0.5)
    )
    d = det.postprocess(boxes, logits, 100, 100)
    np.testing.assert_allclose(d.score, [0.8, 0.7, 0.6], rtol=1e-6)
    assert d.cls.tolist() == [0, 1, 2]


def test_postprocess_floor_and_clipping():
    det = _bare(score_floor=0.5, cls=RFDETROnnx)  # RF-DETR clips to the image
    boxes = np.array([[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 0.1, 0.1]], np.float32)
    logits = np.full((2, 3), -20.0, np.float32)
    logits[0, 1] = _logit(0.7)
    logits[1, 2] = _logit(0.4)  # below the floor
    d = det.postprocess(boxes, logits, 100, 200)
    assert len(d.score) == 1
    assert d.xyxy[0].tolist() == pytest.approx([0, 0, 40, 20])


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown detector kind"):
        make_detector("nope", "x.onnx", 640, 0.1, 300)
