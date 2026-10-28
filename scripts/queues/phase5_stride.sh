#!/usr/bin/env bash
# Phase 5: accuracy of detection stride (D-610), val only. Three at a time (memory).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs; mkdir -p "$L"
say() { echo "[$(date +%H:%M)] $*" >> $L/phase5.log; }
T="--tuned-from B2_yolo26n_960"
run() { $PY scripts/track_tuned.py "$@" >> $L/phase5_stride.log 2>&1; say "track $* exit=$?"; }
run --dets B2_yolo26n_960w_640 $T --gmc-downscale 8 --tag p5_n960w640_bot8 &
run --dets B2_yolo26n_960 $T --gmc-downscale 8 --det-stride 2 --tag p5_n960_bot8_s2 &
run --dets B2_yolo26n_960 $T --gmc-downscale 8 --det-stride 3 --tag p5_n960_bot8_s3 &
wait
run --dets B2_yolo26n_960w_640 $T --gmc-downscale 8 --det-stride 2 --tag p5_n960w640_bot8_s2 &
run --dets B2_yolo26n_960w_640 $T --gmc-downscale 8 --det-stride 3 --tag p5_n960w640_bot8_s3 &
wait
say "stride accuracy done"
