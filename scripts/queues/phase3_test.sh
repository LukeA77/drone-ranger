#!/usr/bin/env bash
# Phase 3 final test-dev run (once). Usage: scripts/queues/phase3_test.sh <run> <imgsz>
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs; mkdir -p "$L"
RUN=$1; IMGSZ=$2
W=cache/train/$RUN/weights/best.pt
say() { echo "[$(date +%H:%M)] $*" >> $L/phase3_test.log; }
say "start $RUN"
$PY scripts/freeze.py --phase phase3 --check >> $L/phase3_test.log 2>&1 || { say "STOP: frozen settings changed"; exit 1; }
$PY scripts/track_eval.py --run ${RUN}_tuned_botsort_cmc --weights $W --class-map visdrone10 --imgsz $IMGSZ --tracker configs/trackers/botsort_cmc_tuned_$RUN.yaml --split test-dev --final > $L/test_track_$RUN.log 2>&1; say "track exit=$?"
$PY scripts/detect_eval.py --run $RUN --weights $W --class-map visdrone10 --imgsz $IMGSZ --split test-dev --final > $L/test_detect_$RUN.log 2>&1; say "detect exit=$?"
say "done"
