#!/usr/bin/env bash
# Phase 5 latency side of the Pareto study: times each configuration (D-601 protocol).
# Run on a quiet machine, plugged in, with other applications closed.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs; mkdir -p "$L"
W=cache/train/B2_yolo26n_960/weights/best.pt
say() { echo "[$(date +%H:%M)] $*" >> $L/phase5.log; }
NOTE="${NOTE:-quiet machine, plugged in, other applications closed}"
T="--tuned-from B2_yolo26n_960"
b() { tag=$1; shift; $PY scripts/benchmark_pipeline.py --run B2_yolo26n_960 $T --tag $tag --note "$NOTE" "$@" > $L/bench_$tag.log 2>&1; say "bench $tag exit=$? $(tail -n 1 $L/bench_$tag.log | cut -c1-120)"; }
say "bench start"
b p5_pt960_bot2 --weights $W --imgsz 960
b p5_onnx960_bot2 --weights weights/B2_yolo26n_960_rect.onnx --imgsz 960
b p5_int8dyn960_bot2 --weights weights/B2_yolo26n_960_rect_int8dyn.onnx --imgsz 960
b p5_int8st960_bot2 --weights weights/B2_yolo26n_960_rect_int8static_headfp32.onnx --imgsz 960
b p5_onnx960_bot4 --weights weights/B2_yolo26n_960_rect.onnx --imgsz 960 --gmc-downscale 4
b p5_onnx960_bot8 --weights weights/B2_yolo26n_960_rect.onnx --imgsz 960 --gmc-downscale 8
b p5_onnx960_byte --weights weights/B2_yolo26n_960_rect.onnx --imgsz 960 --tracker bytetrack --high 0.35 --new 0.55
b p5_int8dyn960_bot4 --weights weights/B2_yolo26n_960_rect_int8dyn.onnx --imgsz 960 --gmc-downscale 4
b p5_onnx640_bot4 --weights weights/B2_yolo26n_960w_640_rect.onnx --imgsz 640 --gmc-downscale 4
b p5_int8dyn640_bot4 --weights weights/B2_yolo26n_960w_640_rect_int8dyn.onnx --imgsz 640 --gmc-downscale 4
b p5_int8dyn640_byte --weights weights/B2_yolo26n_960w_640_rect_int8dyn.onnx --imgsz 640 --tracker bytetrack --high 0.35 --new 0.55
b p5_pt960_bot2_repeat --weights $W --imgsz 960
say "bench done"
