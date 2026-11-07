#!/usr/bin/env bash
# Score the SynDrone fine-tuned detector, leave one town out: each development town is run
# with the model trained without it (cache/train/syndrone_ft_outputs/, exported to ONNX by
# scripts/export_onnx.py), all under one run name. Then both that run and the zero-shot run
# with the same tracker (v2trk_accurate) are evaluated, so the comparison changes only the
# detector.
# Usage: bash scripts/queues/syndrone_finetuned.sh   (log: cache/logs/ft/queue.log)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs/ft; mkdir -p "$L"
Q=$L/queue.log
NAME=ft_loto_accurate
say() { echo "[$(date '+%F %H:%M')] $*" >> "$Q"; }

say "queue start"
for T in Town01 Town07 Town10HD; do
  W=weights/syndrone_ft_minus_${T}_960_rect.onnx
  for h in 80 50 20; do
    $PY scripts/syndrone_run.py --town "$T" --height "$h" --name "$NAME" --weights "$W" \
      > "$L/run_${T}_h$h.log" 2>&1
    say "run $T h$h exit=$?"
  done
done
for N in v2trk_accurate "$NAME"; do
  $PY scripts/syndrone_evaluate.py --name "$N" --split development > "$L/eval_$N.log" 2>&1
  say "eval $N exit=$?"
done
say "queue done"
