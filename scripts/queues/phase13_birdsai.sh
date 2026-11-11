#!/usr/bin/env bash
# Phase 13 thermal tracking on BIRDSAI (D-754), people only.
#   bash scripts/queues/phase13_birdsai.sh tune            # train videos: cache, tracker grid
#   bash scripts/queues/phase13_birdsai.sh test            # once, each detector with the thresholds
#                                                          # its train grid selected
# The thermal detector (weights/dfine_hituav_day_coco.onnx) and, for transfer, the colour one
# (weights/dfine_n_640.onnx), each with the project's tracker (configs/trackers/motion.yaml)
# tuned on BIRDSAI train for people by the D-733 rule (highest F2), detection every frame.
# RUNS="a b" overrides the detectors (default: the thermal and the colour one).
# Log: cache/logs/phase13/birdsai.log
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs/phase13; mkdir -p "$L"
# One Phase 13 queue at a time: two at once ran out of memory (15 GB laptop).
mkdir "$L/.lock" 2> /dev/null || { echo "another Phase 13 queue is running ($L/.lock)"; exit 1; }
trap 'rmdir "$L/.lock"' EXIT
say() { echo "[$(date '+%F %H:%M')] $*" >> "$L/birdsai.log"; }
WORKERS=${WORKERS:-5}
HIGH_GRID="0.05 0.15 0.25 0.35 0.45 0.55"
NEW_GRID="0.25 0.35 0.45 0.55 0.65 0.75 0.85"
STAGE=$1
[ -f cache/birdsai/mot/train/0000000010_0000000000/seqinfo.json ] || "$PY" scripts/birdsai_prep.py
for RUN in ${RUNS:-dfine_hituav_day_coco dfine_n_640}; do
  W=weights/$RUN.onnx
  if [ "$STAGE" = tune ]; then
    "$PY" scripts/cache_detections.py --dataset birdsai --split train --run "$RUN" --weights "$W" \
      --kind dfine --class-map visdrone10 --imgsz 640 > "$L/birdsai_dets_train_$RUN.log" 2>&1
    say "dets train $RUN exit=$?"
    # shellcheck disable=SC2086
    "$PY" scripts/track_grid.py --dataset birdsai --dets "$RUN" --class-map visdrone10 \
      --tracker configs/trackers/motion.yaml --group person --workers "$WORKERS" \
      --high $HIGH_GRID --new $NEW_GRID > "$L/birdsai_grid_$RUN.log" 2>&1
    say "grid train $RUN exit=$?"
  elif [ "$STAGE" = test ]; then
    "$PY" scripts/cache_detections.py --dataset birdsai --split test --final --run "$RUN" \
      --weights "$W" --kind dfine --class-map visdrone10 --imgsz 640 > "$L/birdsai_dets_test_$RUN.log" 2>&1
    say "dets test $RUN exit=$?"
    "$PY" scripts/detect_eval.py --dataset birdsai --split test --final --run "$RUN" --weights "$W" \
      --kind dfine --class-map visdrone10 --imgsz 640 > "$L/birdsai_detect_test_$RUN.log" 2>&1
    say "detect test $RUN exit=$?"
    SEL=$("$PY" -c "import json,sys; s=json.load(open(sys.argv[1]))['selected']; print(s['track_high_thresh'], s['new_track_thresh'])"       "results/birdsai/train/grid_${RUN}__motion__person.json") || { say "no train grid for $RUN"; continue; }
    read -r H N <<< "$SEL"
    "$PY" scripts/track_grid.py --dataset birdsai --split test --final --dets "$RUN"       --class-map visdrone10 --tracker configs/trackers/motion.yaml --group person --workers 1       --high "$H" --new "$N" --suffix _test > "$L/birdsai_track_test_$RUN.log" 2>&1
    say "track test $RUN (high $H new $N) exit=$?"
  else
    echo "stage: tune or test"; exit 1
  fi
done
say "$STAGE done"
