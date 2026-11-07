"""One way to build a detector, whatever its kind, for the pipeline and the scoring scripts.

`make_detector(kind, weights, ...)` returns a function from a BGR image to
(boxes as left, top, width, height; scores; class indices in VisDrone training order, or COCO
order for COCO-trained YOLO weights). Kinds:
- "yolo": the version 1 detector as an ONNX export, run on ONNX Runtime alone
  (`onnx_detectors.YoloOnnx`, which reproduces the training library's predict path: 100 %
  of detections matched, scripts/check_yolo_onnx.py, D-721). PyTorch checkpoints (.pt)
  need that library (Ultralytics, AGPL-3.0), which the package does not use; version 1's
  scripts load them through scripts/ultralytics_v1.py;
- "rfdetr", "dfine": an RF-DETR or D-FINE candidate exported to ONNX
  (`onnx_detectors.RFDETROnnx`, `DFINEOnnx`), run with ONNX Runtime alone. The input size is
  fixed by the export; `imgsz` must match it.

`score_floor=None` keeps the training library's predict default for "yolo" (0.25,
YOLO_PREDICT_CONF), which is what version 1's detection scores used; scoring scripts pass the
floor explicitly so every candidate is scored with the same one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Detect = Callable[[NDArray[np.uint8]], tuple[NDArray, NDArray, NDArray]]
KINDS = ("yolo", "rfdetr", "dfine")
# Version 1's training library (Ultralytics 8.4, engine/predictor.py): `conf` falls
# back to 0.25 when not given.
YOLO_PREDICT_CONF = 0.25
# Detections kept per image by both libraries: rfdetr 1.12 models/postprocess.py
# (num_select=300) and D-FINE configs (num_top_queries: 300).
DETR_NUM_SELECT = 300


def make_detector(
    kind: str,
    weights: str | Path,
    imgsz: int,
    score_floor: float | None,
    max_det: int,
    classes: Sequence[int] | None = None,
) -> Detect:
    """Build a detector. `classes` keeps only those class indices (None: all)."""
    keep = None if classes is None else np.asarray(sorted(classes), int)
    if kind == "yolo" and Path(weights).suffix == ".onnx":
        from aerialrange.detect.onnx_detectors import YoloOnnx

        yolo = YoloOnnx(weights, YOLO_PREDICT_CONF if score_floor is None else score_floor, max_det)
        if imgsz not in (yolo.w, yolo.h):
            raise ValueError(f"{weights}: exported for {yolo.w} px input, asked for {imgsz}")

        def detect_yolo_onnx(img: NDArray[np.uint8]) -> tuple[NDArray, NDArray, NDArray]:
            d = yolo(img)
            m = np.ones(len(d.cls), bool) if keep is None else np.isin(d.cls, keep)
            x = d.xyxy[m]
            return np.column_stack([x[:, :2], x[:, 2:] - x[:, :2]]), d.score[m], d.cls[m]

        return detect_yolo_onnx
    if kind == "yolo":
        raise ValueError(
            f"{weights}: the package runs ONNX models only; export with scripts/export_onnx.py "
            "or load PyTorch checkpoints through scripts/ultralytics_v1.py"
        )
    if kind in ("rfdetr", "dfine"):
        from aerialrange.detect.onnx_detectors import DFINEOnnx, RFDETROnnx

        floor = YOLO_PREDICT_CONF if score_floor is None else float(score_floor)
        cls = RFDETROnnx if kind == "rfdetr" else DFINEOnnx
        det = cls(weights, score_floor=floor, max_det=min(max_det, DETR_NUM_SELECT))
        if imgsz not in (det.w, det.h):
            raise ValueError(f"{weights}: exported for {det.w} px input, asked for {imgsz}")

        def detect_detr(img: NDArray[np.uint8]) -> tuple[NDArray, NDArray, NDArray]:
            d = det(img)
            m = np.ones(len(d.cls), bool) if keep is None else np.isin(d.cls, keep)
            x = d.xyxy[m]
            return np.column_stack([x[:, :2], x[:, 2:] - x[:, :2]]), d.score[m], d.cls[m]

        return detect_detr
    raise ValueError(f"unknown detector kind {kind!r}; expected one of {KINDS}")


def library_version(kind: str) -> dict[str, str]:
    """Versions of what runs a detector of this kind, for result files."""
    from importlib.metadata import version

    return {"onnxruntime": version("onnxruntime")}
