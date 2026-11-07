#!/usr/bin/env bash
# Download the SynDrone towns not yet cached, build their truth, and run the pipeline on the
# calibration towns; re-run the development towns with the project's tracker (D-713).
# Test towns get truth only: their pipeline runs happen once, after freezing.
# Usage: bash scripts/queues/syndrone_remaining.sh   (log: cache/logs/remaining/queue.log)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs/remaining; mkdir -p "$L"
Q=$L/queue.log
NAME=v2trk_accurate  # version 1 detector, the project's tracker
say() { echo "[$(date '+%F %H:%M')] $*" >> "$Q"; }

fetch() {  # retried: a rerun skips members already verified and written
  local T=$1 rc=1
  for attempt in 1 2 3 4; do
    $PY scripts/syndrone_fetch.py --town "$T" --modality camera bboxes rgb semantic depth \
      >> "$L/fetch_$T.log" 2>&1
    rc=$?; [ $rc -eq 0 ] && break
    say "fetch $T attempt $attempt exit=$rc; retrying in 60 s"; sleep 60
  done
  say "fetch $T exit=$rc"
}

wait_fetch() {  # until this town's fetch has finished (any exit)
  until grep -q "fetch $1 exit=" "$Q" 2>/dev/null; do sleep 60; done
  grep -q "fetch $1 exit=0" "$Q"
}

truth() {
  local T=$1
  $PY scripts/syndrone_actor_tracks.py --town "$T" > "$L/actors_$T.log" 2>&1; say "actors $T exit=$?"
  $PY scripts/syndrone_speed_truth.py --town "$T" > "$L/speed_truth_$T.log" 2>&1; say "speed truth $T exit=$?"
  $PY scripts/syndrone_truth.py --town "$T" --workers 5 > "$L/truth_$T.log" 2>&1; say "truth $T exit=$?"
  $PY scripts/syndrone_z0_check.py --town "$T" --sheet > "$L/z0_$T.log" 2>&1; say "z0 check $T exit=$?"
}

runs() {
  local T=$1
  for h in 80 50 20; do
    $PY scripts/syndrone_run.py --town "$T" --height "$h" --name "$NAME" \
      > "$L/run_${T}_h$h.log" 2>&1
    say "run $T h$h exit=$?"
  done
}

say "queue start"
# download lanes (network bound)
( fetch Town04; fetch Town02; fetch Town06 ) &
( fetch Town05; fetch Town03 ) &
# processing lane (CPU bound), one job at a time
(
  for T in Town01 Town07 Town10HD; do runs "$T"; done
  for T in Town04 Town05; do
    if wait_fetch "$T"; then truth "$T"; runs "$T"; else say "skip $T: fetch failed"; fi
  done
  for T in Town02 Town03 Town06; do
    if wait_fetch "$T"; then truth "$T"; else say "skip $T: fetch failed"; fi
  done
) &
wait
.venv/Scripts/python.exe scripts/check_data.py > "$L/check_data.log" 2>&1; say "check_data exit=$?"
say "queue done"
