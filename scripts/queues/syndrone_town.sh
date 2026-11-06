#!/usr/bin/env bash
# Phase 7 per-town queue (development towns only): truth, checks, then the pipeline on all
# three cameras. Usage: bash scripts/queues/syndrone_town.sh Town07
set -euo pipefail
T="$1"
PY=.venv/Scripts/python.exe
L=cache/logs
$PY scripts/syndrone_actor_tracks.py --town "$T" > "$L/syndrone_actors_$T.log" 2>&1
$PY scripts/syndrone_speed_truth.py --town "$T" > "$L/syndrone_speed_truth_$T.log" 2>&1
$PY scripts/syndrone_truth.py --town "$T" --workers 5 > "$L/syndrone_truth_$T.log" 2>&1
$PY scripts/syndrone_z0_check.py --town "$T" --sheet > "$L/syndrone_z0_$T.log" 2>&1
for h in 80 50 20; do
  $PY scripts/syndrone_run.py --town "$T" --height "$h" > "$L/syndrone_run_${T}_h$h.log" 2>&1
done
echo "done $T"
