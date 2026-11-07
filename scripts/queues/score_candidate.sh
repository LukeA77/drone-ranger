#!/usr/bin/env bash
# Score one trained detector candidate on VisDrone validation, start to finish.
# Usage: bash scripts/queues/score_candidate.sh <run> <imgsz> <kind>
#   e.g. rfdetr_nano_640 640 rfdetr  |  dfine_n_640 640 dfine
# Needs cache/train/<run>/ (the Kaggle output, unzipped).
#  1. export to ONNX and check parity with the training code (its own environment, D-717);
#  2. detection AP on MOT val frames, same protocol and score floor as version 1;
#  3. detections cached for every MOT val frame;
#  4. the project's tracker tuned on those detections with version 1's threshold grid and
#     selection rule (D-319, configs/tracker.yaml), so each detector gets its best tracker.
# Log: cache/logs/candidates/<run>.log
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
RUN=$1; IMGSZ=$2; KIND=$3; WORKERS=${WORKERS:-5}  # about 1 GB of RAM per worker
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
ENV_PY() { local p=cache/venvs/$1/Scripts/python.exe; [ -x "$p" ] || p=cache/venvs/$1/bin/python; echo "$p"; }
L=cache/logs/candidates; mkdir -p "$L"
Q=$L/$RUN.log
say() { echo "[$(date '+%F %H:%M')] $*" >> "$Q"; }
HIGH="0.05 0.15 0.25 0.35 0.45 0.55"
NEW="0.25 0.35 0.45 0.55 0.65 0.75 0.85"

say "start $RUN"
case $KIND in
  rfdetr) PYTHONUTF8=1 "$(ENV_PY rfdetr)" scripts/export_rfdetr.py --run "$RUN" ;;
  dfine) PYTHONUTF8=1 "$(ENV_PY dfine)" scripts/export_dfine.py --run "$RUN" --repo cache/venvs/src/D-FINE ;;
  *) echo "unknown kind $KIND"; false ;;
esac > "$L/${RUN}_export.log" 2>&1
rc=$?; say "export exit=$rc"; [ $rc -eq 0 ] || exit $rc
W=weights/$RUN.onnx
$PY scripts/detect_eval.py --run "$RUN" --weights "$W" --kind "$KIND" --class-map visdrone10 \
  --imgsz "$IMGSZ" > "$L/${RUN}_detect.log" 2>&1
say "detect exit=$?"
$PY scripts/cache_detections.py --run "$RUN" --weights "$W" --kind "$KIND" \
  --class-map visdrone10 --imgsz "$IMGSZ" > "$L/${RUN}_dets.log" 2>&1
rc=$?; say "dets exit=$rc"; [ $rc -eq 0 ] || exit $rc
# shellcheck disable=SC2086
$PY scripts/track_grid.py --dets "$RUN" --class-map visdrone10 \
  --tracker configs/trackers/motion.yaml --workers "$WORKERS" --high $HIGH --new $NEW > "$L/${RUN}_grid.log" 2>&1
say "grid exit=$?"
say "done $RUN"
