"""DETR-style detectors run with ONNX Runtime alone (no training library at run time).

RF-DETR (Apache-2.0, scripts/export_rfdetr.py) and D-FINE (Apache-2.0, scripts/export_dfine.py)
share one decoding, which both libraries use (rfdetr 1.12 `models/postprocess.py`; D-FINE
`src/zoo/dfine/postprocessor.py`): the raw network outputs are boxes (1, Q, 4) as normalised
centre x, centre y, width, height and class logits (1, Q, S); scores are the sigmoid of the
logits, and the `max_det` highest over all query and class pairs are kept, each with its
query's box scaled to the image. The export check compares this module with each library.

Input: RGB, resized to the model's square input with PIL's bilinear resize (which
antialiases when shrinking), scaled to [0, 1]. That is the resize both libraries' validation
pipelines apply (torchvision `Resize` on PIL images), so the model sees what it was selected
on. RF-DETR then normalises with the ImageNet mean and standard deviation; D-FINE does not.
(RF-DETR's quick `predict()` resizes without antialiasing; the validation path is the
reference, D-719.) RF-DETR clips boxes to the image, D-FINE does not; each is reproduced.

Class slots: the VisDrone package uses the VisDrone category numbers 1..10 as COCO category
ids (scripts/visdrone_coco_package.py). RF-DETR renumbers a dataset's categories from 0 when
it trains (category 1 is label 0; its own class list reads the same way), so its slot s is
training class s. D-FINE trains on the ids as they are, so its slot s is training class
s - 1 and slot 0 is unused. Either way the result is the version 1 detector's class index
(VisDrone order, 0..9); slots outside the ten classes are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class Detections:
    xyxy: NDArray[np.float64]  # (N, 4) pixels in the input image
    score: NDArray[np.float64]  # (N,)
    cls: NDArray[np.int64]  # (N,) training class index (VisDrone order, 0..9)

    @property
    def xywh(self) -> NDArray[np.float64]:
        """Centre x, centre y, width, height."""
        x = self.xyxy
        return np.column_stack(
            [(x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2, x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]]
        )


def _sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-x))


class DetrOnnx:
    """A DETR-style detector exported to ONNX (fixed input size, batch 1)."""

    output_names: tuple[str, str] = ("boxes", "logits")  # (boxes, class logits)
    normalize: bool = False
    clip: bool = False
    first_slot: int = 1  # output slot of training class 0
    n_classes: int = 10

    def __init__(
        self,
        path: str | Path,
        score_floor: float,
        max_det: int = 300,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = (int(v) for v in inp.shape)
        names = [o.name for o in self.session.get_outputs()]
        if not set(self.output_names) <= set(names):
            raise ValueError(f"{path}: outputs {names}, expected {list(self.output_names)}")
        self.score_floor = float(score_floor)
        self.max_det = int(max_det)

    def preprocess(self, img_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        x = np.asarray(rgb.resize((self.w, self.h), Image.Resampling.BILINEAR), np.float32)
        x = x / 255.0
        if self.normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)

    def postprocess(self, boxes: NDArray, logits: NDArray, img_h: int, img_w: int) -> Detections:
        _, s = logits.shape
        prob = _sigmoid(logits.astype(np.float64)).ravel()
        k = min(self.max_det, prob.size)
        top = np.argpartition(-prob, k - 1)[:k]
        top = top[np.argsort(-prob[top], kind="stable")]
        score = prob[top]
        slot = top % s
        query = top // s
        cls = slot - self.first_slot
        keep = (score > self.score_floor) & (cls >= 0) & (cls < self.n_classes)
        score, cls, query = score[keep], cls[keep], query[keep]
        b = boxes[query].astype(np.float64)
        xyxy = np.column_stack(
            [
                b[:, 0] - b[:, 2] / 2,
                b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2,
                b[:, 1] + b[:, 3] / 2,
            ]
        ) * np.array([img_w, img_h, img_w, img_h], np.float64)
        if self.clip:
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)
        return Detections(xyxy, score, cls.astype(np.int64))

    def __call__(self, img_bgr: NDArray[np.uint8]) -> Detections:
        out = self.session.run(list(self.output_names), {self.input_name: self.preprocess(img_bgr)})
        return self.postprocess(out[0][0], out[1][0], img_bgr.shape[0], img_bgr.shape[1])


class RFDETROnnx(DetrOnnx):
    """RF-DETR's own ONNX export (outputs `dets`, `labels`)."""

    output_names = ("dets", "labels")
    normalize = True
    clip = True
    first_slot = 0


class DFINEOnnx(DetrOnnx):
    """D-FINE network without its post-processor (scripts/export_dfine.py)."""

    output_names = ("boxes", "logits")
    normalize = False
    clip = False
    first_slot = 1


def nms_xyxy(boxes: NDArray, scores: NDArray, iou_thres: float) -> NDArray[np.int64]:
    """Greedy non-maximum suppression, highest score first (as torchvision.ops.nms)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, np.int64)


class YoloOnnx:
    """The version 1 detector (YOLO26n, Ultralytics export without built-in NMS) on ONNX
    Runtime alone, reproducing the library's predict path (Ultralytics 8.4.182):
    letterbox to the export's input size (bilinear resize keeping the aspect ratio, padding
    with grey 114, centred; `data/augment.py` LetterBox), BGR to RGB, scaled to [0, 1]; output
    (1, 4 + classes, anchors) as centre x, centre y, width, height in input pixels and class
    scores; best class per box above the score floor, class-aware NMS at IoU 0.7 (the
    library's predict default; boxes offset by class x 7680, `utils/nms.py`), at most
    `max_det` kept, then mapped back to the frame and clipped (`utils/ops.py` scale_boxes).
    Checked against the library by scripts/check_yolo_onnx.py."""

    IOU = 0.7
    MAX_WH = 7680
    MAX_NMS = 30000

    def __init__(
        self, path: str | Path, score_floor: float, max_det: int = 300, threads: int | None = None
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = (int(v) for v in inp.shape)
        self.score_floor = float(score_floor)
        self.max_det = int(max_det)

    def letterbox(self, img_bgr: NDArray[np.uint8]) -> tuple[NDArray[np.float32], tuple]:
        h0, w0 = img_bgr.shape[:2]
        r = min(self.h / h0, self.w / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        im = img_bgr
        if (w0, h0) != (nw, nh):
            im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (self.w - nw) / 2, (self.h - nh) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,) * 3)
        x = im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), (nw / w0, nh / h0, left, top)

    def postprocess(self, out: NDArray, geom: tuple, img_h: int, img_w: int) -> Detections:
        gx, gy, px, py = geom
        p = out.T  # (anchors, 4 + classes)
        cls_scores = p[:, 4:]
        j = cls_scores.argmax(1)
        conf = cls_scores[np.arange(len(j)), j]
        m = conf > self.score_floor
        p, conf, j = p[m], conf[m], j[m]
        if len(conf) > self.MAX_NMS:
            top = np.argsort(-conf, kind="stable")[: self.MAX_NMS]
            p, conf, j = p[top], conf[top], j[top]
        b = p[:, :4].astype(np.float64)
        xyxy = np.column_stack(
            [
                b[:, 0] - b[:, 2] / 2,
                b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2,
                b[:, 1] + b[:, 3] / 2,
            ]
        )
        keep = nms_xyxy(xyxy + (j * self.MAX_WH)[:, None], conf.astype(np.float64), self.IOU)
        keep = keep[: self.max_det]
        xyxy, conf, j = xyxy[keep], conf[keep], j[keep]
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / gx).clip(0, img_w)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / gy).clip(0, img_h)
        return Detections(xyxy, conf.astype(np.float64), j.astype(np.int64))

    def __call__(self, img_bgr: NDArray[np.uint8]) -> Detections:
        x, geom = self.letterbox(img_bgr)
        out = self.session.run(None, {self.input_name: x})[0][0]
        return self.postprocess(out, geom, img_bgr.shape[0], img_bgr.shape[1])
