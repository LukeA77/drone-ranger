#!/usr/bin/env bash
# Phase 5 accuracy side of the Pareto study (val only). No timing here.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs; mkdir -p "$L"
W=cache/train/B2_yolo26n_960/weights/best.pt
say() { echo "[$(date +%H:%M)] $*" >> $L/phase5.log; }
say "accuracy start"
# n960 weights run at 640 input (smaller, faster)
$PY scripts/export_onnx.py --weights $W --imgsz 640 --name B2_yolo26n_960w > $L/export_960w_640.log 2>&1; say "export 640 exit=$?"
$PY scripts/quantize_onnx.py --onnx weights/B2_yolo26n_960w_640_rect.onnx > $L/quant_640.log 2>&1; say "quant 640 exit=$?"
$PY scripts/detect_eval.py --run B2_yolo26n_960w_640 --weights $W --class-map visdrone10 --imgsz 640 > $L/detect_960w_640.log 2>&1; say "detect 960w@640 exit=$?"
$PY scripts/detect_eval.py --run B2_yolo26n_960w_640_int8dyn --weights weights/B2_yolo26n_960w_640_rect_int8dyn.onnx --class-map visdrone10 --imgsz 640 > $L/detect_960w_640_int8dyn.log 2>&1; say "detect 960w@640 int8dyn exit=$?"
# detection caches (score floor 0.1) for tracking accuracy
for spec in "B2_yolo26n_960_int8dyn weights/B2_yolo26n_960_rect_int8dyn.onnx 960" "B2_yolo26n_960_int8st weights/B2_yolo26n_960_rect_int8static_headfp32.onnx 960" "B2_yolo26n_960w_640 $W 640" "B2_yolo26n_960w_640_int8dyn weights/B2_yolo26n_960w_640_rect_int8dyn.onnx 640"; do
  set -- $spec
  $PY scripts/cache_detections.py --run $1 --weights $2 --class-map visdrone10 --imgsz $3 > $L/dets_$1.log 2>&1; say "dets $1 exit=$?"
done
# tracking accuracy on val (tuned thresholds of B2 n960, D-321; not re-tuned per variant)
T="--tuned-from B2_yolo26n_960"
run() { $PY scripts/track_tuned.py "$@" >> $L/phase5_track.log 2>&1; say "track $* exit=$?"; }
run --dets B2_yolo26n_960 $T --gmc-downscale 4 --tag p5_n960_bot4
run --dets B2_yolo26n_960 $T --gmc-downscale 8 --tag p5_n960_bot8
run --dets B2_yolo26n_960 $T --tracker bytetrack --high 0.35 --new 0.55 --tag p5_n960_byte
run --dets B2_yolo26n_960_int8dyn $T --tag p5_int8dyn960_bot2
run --dets B2_yolo26n_960_int8dyn $T --gmc-downscale 4 --tag p5_int8dyn960_bot4
run --dets B2_yolo26n_960_int8st $T --tag p5_int8st960_bot2
run --dets B2_yolo26n_960w_640 $T --gmc-downscale 4 --tag p5_n960w640_bot4
run --dets B2_yolo26n_960w_640_int8dyn $T --gmc-downscale 4 --tag p5_int8dyn640_bot4
run --dets B2_yolo26n_960w_640_int8dyn $T --tracker bytetrack --high 0.35 --new 0.55 --tag p5_int8dyn640_byte
say "accuracy done"
