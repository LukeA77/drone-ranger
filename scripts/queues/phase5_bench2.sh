#!/usr/bin/env bash
# Phase 5 round 2 (D-610): detection stride and overlapped pipeline. Quiet machine.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs; mkdir -p "$L"
say() { echo "[$(date +%H:%M)] $*" >> $L/phase5.log; }

NOTE="${NOTE:-quiet machine, plugged in, other applications closed; background load measured per run}"
O640=weights/B2_yolo26n_960w_640_rect.onnx
O960=weights/B2_yolo26n_960_rect.onnx
seq() { tag=$1; shift; $PY scripts/benchmark_pipeline.py --run B2_yolo26n_960 --tuned-from B2_yolo26n_960 --tag $tag --note "$NOTE" "$@" > $L/bench_$tag.log 2>&1; say "bench $tag exit=$? $(tail -n 1 $L/bench_$tag.log | cut -c1-110)"; }
mt() { tag=$1; shift; $PY scripts/benchmark_threaded.py --tag $tag --note "$NOTE" "$@" > $L/bench_$tag.log 2>&1; say "bench $tag exit=$? $(tail -n 1 $L/bench_$tag.log | cut -c1-110)"; }
say "bench2 start"
for s in 1 2 3; do
  seq p5_onnx640_bot8_s$s --weights $O640 --imgsz 640 --gmc-downscale 8 --det-stride $s
  mt p5_mt_onnx640_bot8_s$s --weights $O640 --imgsz 640 --gmc-downscale 8 --det-stride $s
  mt p5_mt_onnx960_bot8_s$s --weights $O960 --imgsz 960 --gmc-downscale 8 --det-stride $s
done
seq p5_onnx960_bot8_s2 --weights $O960 --imgsz 960 --gmc-downscale 8 --det-stride 2
seq p5_onnx960_bot8_s3 --weights $O960 --imgsz 960 --gmc-downscale 8 --det-stride 3
say "bench2 done"
