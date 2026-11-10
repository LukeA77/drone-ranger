#!/usr/bin/env bash
# Phase 10 (build2 Section 9, D-749): the one-time end-to-end test on the SynDrone test towns.
# Runs the frozen version 2 pipeline (configs/run.yaml: D-FINE N, own tracker, range and speed
# stage) once on every test town and height, applies the interval calibration fitted on the
# calibration towns, evaluates with bootstrap intervals over sequences, breaks the results
# down by condition and reports false alarms per km2 searched and the share of people found;
# then repeats the ranging and speed part with version 1's three navigation grades added to the
# camera poses (fixed seeds), tracks unchanged.
#
# Usage: bash scripts/queues/phase10_syndrone_test.sh          the test run (once, after freezing)
#        SPLIT=development FRAMES=10 NAME=zz_p10_check bash scripts/queues/phase10_syndrone_test.sh
#                                                            a short check of the chain on
#                                                            development towns (not a result)
# The test run first checks the freeze (scripts/freeze.py --phase phase10 --check) and stops if
# any frozen file changed. It refuses to run if its run folder already exists, so the test towns
# cannot be run twice by accident. Log: cache/logs/phase10/<name>/queue.log
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
SPLIT=${SPLIT:-test}
NAME=${NAME:-p10_v2}
FRAMES=${FRAMES:-}
CAL=results/syndrone/calibration/interval_calibration_v2_dfine.json
GRADES="nav_good nav_typical nav_poor"
L=cache/logs/phase10/$NAME; mkdir -p "$L"
Q=$L/queue.log
say() { echo "[$(date '+%F %H:%M')] $*" | tee -a "$Q"; }
step() {  # step <label> <log> <command...>
  local label=$1 log=$2; shift 2
  "$@" > "$log" 2>&1; local rc=$?
  say "$label exit=$rc"
  if [ $rc -ne 0 ]; then say "STOPPED: see $log"; exit $rc; fi
}

TOWNS=$("$PY" -c "import json; print(' '.join(json.load(open('results/splits.json'))['syndrone']['splits']['$SPLIT']))")
ALLOW=()
if [ "$SPLIT" = test ]; then
  [ -z "$FRAMES" ] || { echo "FRAMES is for development checks only"; exit 2; }
  ALLOW=(--allow-test)
  step "freeze check" "$L/freeze_check.log" "$PY" scripts/freeze.py --phase phase10 --check
fi
if [ -e "cache/syndrone/runs/$NAME" ]; then
  say "cache/syndrone/runs/$NAME exists: not run again (a test run is made once)"; exit 3
fi
say "queue start ($NAME, split $SPLIT, towns $TOWNS${FRAMES:+, $FRAMES frames})"

# 1. detect, track, range and speed, once per test sequence (exact poses)
for T in $TOWNS; do
  for h in 80 50 20; do
    step "run $T h$h" "$L/run_${T}_h$h.log" \
      "$PY" scripts/syndrone_run.py --town "$T" --height "$h" --config run --name "$NAME" \
      "${ALLOW[@]}" ${FRAMES:+--frames "$FRAMES"}
  done
done

# 2. the calibrated intervals (fitted on the calibration towns), tracks unchanged
step "intervals" "$L/intervals.log" \
  "$PY" scripts/syndrone_refilter.py --source "$NAME" --name "${NAME}_cal" --config run \
  --intervals "$CAL" --towns $TOWNS "${ALLOW[@]}"

# 3. evaluation and breakdowns
step "eval ${NAME}_cal" "$L/eval.log" \
  "$PY" scripts/syndrone_evaluate.py --name "${NAME}_cal" --split "$SPLIT" "${ALLOW[@]}"
step "breakdowns ${NAME}_cal" "$L/breakdowns.log" \
  "$PY" scripts/syndrone_breakdowns.py --name "${NAME}_cal" --split "$SPLIT"
step "search ${NAME}_cal" "$L/search.log" \
  "$PY" scripts/syndrone_search_metrics.py --name "${NAME}_cal" --split "$SPLIT"

# 4. the same tracks with version 1's navigation grades added to the poses
for G in $GRADES; do
  step "replay $G" "$L/replay_$G.log" \
    "$PY" scripts/syndrone_refilter.py --source "$NAME" --name "${NAME}_cal_$G" --config run \
    --intervals "$CAL" --nav-grade "$G" --seed 0 --towns $TOWNS "${ALLOW[@]}"
  step "eval $G" "$L/eval_$G.log" \
    "$PY" scripts/syndrone_evaluate.py --name "${NAME}_cal_$G" --split "$SPLIT" "${ALLOW[@]}"
  step "breakdowns $G" "$L/breakdowns_$G.log" \
    "$PY" scripts/syndrone_breakdowns.py --name "${NAME}_cal_$G" --split "$SPLIT"
done
say "queue done"
