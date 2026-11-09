#!/usr/bin/env bash
# Score the SynDrone fine-tuned version 2 detector (D-740), leave one town out, as version 1's
# scripts/queues/syndrone_finetuned.sh: each development town is run with the D-FINE model
# trained without it, all under one run name, with the version 2 configuration (configs/run.yaml);
# then evaluated and compared with the zero-shot run v2_dfine on the same sequences, so the
# comparison changes only the detector's weights.
# Needs cache/train/dfine_syndrone_minus_<Town>/ (the unzipped GPU outputs,
# gpu/train_dfine_syndrone_kaggle.py). Stops at the first failed step.
# Usage: bash scripts/queues/syndrone_finetuned_dfine.sh   (log: cache/logs/ft_dfine/queue.log)
#        FRAMES=10 NAME=zz_check ... for a short check of the chain (not a result)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
PYD=cache/venvs/dfine/Scripts/python.exe; [ -x "$PYD" ] || PYD=cache/venvs/dfine/bin/python
L=cache/logs/ft_dfine; mkdir -p "$L"
Q=$L/queue.log
NAME=${NAME:-ft_dfine_loto}
FRAMES=${FRAMES:-}
TOWNS=${TOWNS:-"Town01 Town07 Town10HD"}
say() { echo "[$(date '+%F %H:%M')] $*" | tee -a "$Q"; }
step() {  # step <label> <log> <command...>
  local label=$1 log=$2; shift 2
  "$@" > "$log" 2>&1; local rc=$?
  say "$label exit=$rc"
  if [ $rc -ne 0 ]; then say "STOPPED: see $log"; exit $rc; fi
}

say "queue start ($NAME)"
for T in $TOWNS; do
  R=dfine_syndrone_minus_$T
  [ -f "cache/train/$R/checkpoint_best.pth" ] || { say "missing cache/train/$R/checkpoint_best.pth"; exit 1; }
  # always exported, so a stale weights/$R.onnx is never scored
  step "export $R" "$L/export_$T.log" \
    "$PYD" scripts/export_dfine.py --run "$R" --repo cache/venvs/src/D-FINE
done
for T in $TOWNS; do
  for h in 80 50 20; do
    step "run $T h$h" "$L/run_${T}_h$h.log" \
      "$PY" scripts/syndrone_run.py --town "$T" --height "$h" --config run --name "$NAME" \
      --weights "weights/dfine_syndrone_minus_$T.onnx" --kind dfine ${FRAMES:+--frames "$FRAMES"}
  done
done
if [ -n "$FRAMES" ]; then say "short check done (no evaluation)"; exit 0; fi
step "eval $NAME" "$L/eval_$NAME.log" \
  "$PY" scripts/syndrone_evaluate.py --name "$NAME" --split development
step "compare v2_dfine vs $NAME" "$L/compare.log" \
  "$PY" scripts/syndrone_compare.py --base v2_dfine --new "$NAME" --split development
say "queue done"
