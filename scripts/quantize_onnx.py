"""INT8 quantisation of the exported detector with ONNX Runtime (D-608).

Usage: python scripts/quantize_onnx.py --onnx weights/best_960_rect.onnx [--n-calib 100]

Writes, next to the FP32 model: `<stem>_int8dyn.onnx` (dynamic: INT8 weights, activations
quantised at run time; no calibration data) and `<stem>_int8static.onnx` (static, QDQ format,
per-channel INT8 weights and UINT8 activations, calibrated on `--n-calib` MOT val frames drawn
with the project seed). Calibration frames get the same letterbox the library applies at
inference (to the model's fixed input shape) and the same scaling to [0, 1]. Validation frames
only; no test data. Accuracy and speed of each variant are measured separately
(detect_eval.py, track_eval.py, benchmark_pipeline.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from aerialrange.utils.config import REPO_ROOT, load_config

PREP = REPO_ROOT / "cache" / "visdrone" / "mot" / "val"


def letterbox(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize keeping the aspect ratio and pad to (h, w) with grey 114, centred, as the library
    does; returns a (1, 3, h, w) float32 RGB tensor in [0, 1]."""
    import cv2

    r = min(h / img.shape[0], w / img.shape[1])
    nh, nw = round(img.shape[0] * r), round(img.shape[1] * r)
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (h - nh) // 2, (w - nw) // 2
    out = np.full((h, w, 3), 114, np.uint8)
    out[top : top + nh, left : left + nw] = im
    return out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument(
        "--head-fp32",
        default=None,
        help="node-name part of the detection head to keep in FP32 in the static model, "
        "e.g. model.23/ (D-608: quantising the head's box and score outputs destroys them)",
    )
    args = ap.parse_args(argv)
    import cv2
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = REPO_ROOT / args.onnx
    model = onnx.load(str(src))
    inp = model.graph.input[0]
    _, _, h, w = (d.dim_value for d in inp.type.tensor_type.shape.dim)
    pre = src.with_name(src.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre))

    frames = []
    for seq in sorted(p for p in PREP.iterdir() if p.is_dir()):
        info = json.loads((seq / "seqinfo.json").read_text())
        frames += sorted((REPO_ROOT / info["frame_dir"]).glob("*.jpg"))
    rng = np.random.default_rng(load_config("project").require("seed"))
    pick = sorted(rng.choice(len(frames), size=min(args.n_calib, len(frames)), replace=False))

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(pick)

        def get_next(self) -> dict | None:
            k = next(self.it, None)
            if k is None:
                return None
            return {inp.name: letterbox(cv2.imread(str(frames[k])), h, w)}

    dyn = src.with_name(src.stem + "_int8dyn.onnx")
    quantize_dynamic(str(pre), str(dyn), weight_type=QuantType.QInt8)
    keep = [n.name for n in model.graph.node if args.head_fp32 and args.head_fp32 in n.name]
    sta = src.with_name(src.stem + ("_int8static_headfp32.onnx" if keep else "_int8static.onnx"))
    quantize_static(
        str(pre),
        str(sta),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        nodes_to_exclude=keep,
    )
    pre.unlink(missing_ok=True)
    rec = {
        "fp32": src.name,
        "input": [int(h), int(w)],
        "int8_dynamic": dyn.name,
        "int8_static": sta.name,
        "static_nodes_kept_fp32": len(keep),
        "calibration_frames": [
            str(Path(frames[k]).relative_to(REPO_ROOT).as_posix()) for k in pick
        ],
        "sizes_mb": {p.name: round(p.stat().st_size / 2**20, 2) for p in (src, dyn, sta)},
    }
    dest = REPO_ROOT / "results" / "edge" / f"quantize_{sta.stem}.json"
    dest.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "calibration_frames"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
