#!/usr/bin/env bash
# Phase 13 one-time thermal test (D-751). Run after phase13_thermal_prepare.sh and the freeze
# (python scripts/freeze.py --phase phase13). Refuses if the freeze has changed or the test
# results exist.
#  1. day-split test days: the thermal detector (from COCO), the one from the VisDrone
#     detector, and the colour (VisDrone) detector, for transfer colour -> thermal;
#  2. the official split's test images: the official-split run (comparison with published
#     results; its test images share videos with its training images);
#  3. thermal -> colour transfer: the thermal detector on VisDrone MOT val (scripts/detect_eval.py).
# Log: cache/logs/phase13/test.log
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs/phase13; mkdir -p "$L"
# One Phase 13 queue at a time: two at once ran out of memory (15 GB laptop).
mkdir "$L/.lock" 2> /dev/null || { echo "another Phase 13 queue is running ($L/.lock)"; exit 1; }
trap 'rmdir "$L/.lock"' EXIT
say() { echo "[$(date '+%F %H:%M')] $*" >> "$L/test.log"; }
"$PY" scripts/freeze.py --phase phase13 --check > "$L/freeze_check.log" 2>&1 || { say "freeze check failed"; exit 1; }
if ls results/thermal/detect_*_test.json > /dev/null 2>&1; then say "test results exist: refusing"; exit 1; fi
say "start"
for RUN in dfine_hituav_day_coco dfine_hituav_day_visdrone; do
  "$PY" scripts/hituav_eval.py --run "$RUN" --weights "weights/$RUN.onnx" --split day \
    --part test --allow-test > "$L/test_$RUN.log" 2>&1; say "day test $RUN exit=$?"
done
"$PY" scripts/hituav_eval.py --run visdrone_dfine_n_640 --weights weights/dfine_n_640.onnx \
  --split day --part test --allow-test > "$L/test_visdrone.log" 2>&1; say "day test visdrone exit=$?"
"$PY" scripts/hituav_eval.py --run dfine_hituav_official_coco \
  --weights weights/dfine_hituav_official_coco.onnx --split official --part test --allow-test \
  > "$L/test_official.log" 2>&1; say "official test exit=$?"
"$PY" scripts/detect_eval.py --run dfine_hituav_day_coco --weights weights/dfine_hituav_day_coco.onnx \
  --class-map visdrone10 --imgsz 640 --kind dfine --split val > "$L/visdrone_val.log" 2>&1
say "thermal on VisDrone val exit=$?"
say "done"
