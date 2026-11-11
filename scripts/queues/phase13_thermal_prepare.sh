#!/usr/bin/env bash
# Phase 13 (D-751), before the freeze: export the three HIT-UAV runs to ONNX with a parity check
# on thermal frames, and score each on its split's validation images only.
# Needs cache/train/<run>/ for each run (the Kaggle zips, unzipped).
# Log: cache/logs/phase13/prepare.log
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
DPY=cache/venvs/dfine/Scripts/python.exe; [ -x "$DPY" ] || DPY=cache/venvs/dfine/bin/python
L=cache/logs/phase13; mkdir -p "$L"
# One Phase 13 queue at a time: two at once ran out of memory (15 GB laptop).
mkdir "$L/.lock" 2> /dev/null || { echo "another Phase 13 queue is running ($L/.lock)"; exit 1; }
trap 'rmdir "$L/.lock"' EXIT
say() { echo "[$(date '+%F %H:%M')] $*" >> "$L/prepare.log"; }
PARITY=Data/hituav/normal_json/val
say "start"
for RUN in dfine_hituav_day_coco dfine_hituav_day_visdrone dfine_hituav_official_coco; do
  if [ ! -f "cache/train/$RUN/checkpoint_best.pth" ]; then say "missing cache/train/$RUN"; continue; fi
  PYTHONUTF8=1 "$DPY" scripts/export_dfine.py --run "$RUN" --repo cache/venvs/src/D-FINE \
    --parity-dir "$PARITY" > "$L/export_$RUN.log" 2>&1
  say "export $RUN exit=$?"
  SPLIT=day; case $RUN in *official*) SPLIT=official ;; esac
  "$PY" scripts/hituav_eval.py --run "$RUN" --weights "weights/$RUN.onnx" --split "$SPLIT" \
    --part val > "$L/val_$RUN.log" 2>&1
  say "val $RUN exit=$?"
done
say "done"
