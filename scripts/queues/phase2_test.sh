#!/usr/bin/env bash
# Phase 2 final test run (once), in batches of 3 test flights (D-606, D-607):
# per batch: download from the Mid-Air site (zip-checked) -> evaluation -> terrain -> release
# (local frames deleted after confirming same-size archive copies). Then the split-level
# summary, analysis and calibration from the per-flight caches.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
L=cache/logs; mkdir -p "$L"
say() { echo "[$(date +%H:%M)] $*" >> $L/phase2_test.log; }
check() { $PY scripts/freeze.py --phase phase2 --check >> $L/phase2_test.log 2>&1 || { say "STOP: frozen settings changed"; exit 1; }; }
say "start (site download)"; check
for batch in "trajectory_0001 trajectory_0005 trajectory_0009" "trajectory_0014 trajectory_0015 trajectory_0017" "trajectory_0022 trajectory_0024 trajectory_0027"; do
  say "batch: $batch"
  for t in $batch; do
    if [ ! -f Data/midair/Kite_training/sunny/depth/$t/.archived ] && [ -f Data/midair/Kite_training/sunny/depth/$t/.frames.done ]; then say "already downloaded $t"; continue; fi
    $PY scripts/archive_midair.py --prepare-download --splits test --trajectories $t >> $L/restore_test.log 2>&1 || { say "STOP: prepare failed"; exit 1; }
    for m in depth segmentation normals color_left; do
      $PY scripts/download_midair.py --only Kite_training/sunny/$m/$t --min-free-gb 6 >> $L/download_test.log 2>&1 || { say "STOP: download $m $t failed"; exit 1; }
    done
    say "downloaded $t"
  done
  check
  $PY scripts/midair_eval.py --split test --final >> $L/test_eval.log 2>&1 || { say "STOP: eval failed"; exit 1; }
  $PY scripts/midair_terrain.py --split test --final >> $L/test_terrain.log 2>&1 || { say "STOP: terrain failed"; exit 1; }
  $PY scripts/archive_midair.py --release --splits test --trajectories $batch >> $L/restore_test.log 2>&1 || { say "STOP: release failed"; exit 1; }
  say "batch done: $batch; free $(df -h . | tail -n 1 | awk '{print $4}')"
done
check
$PY scripts/midair_eval.py --split test --final --from-cache > $L/test_eval_final.log 2>&1; say "eval summary exit=$?"
$PY scripts/midair_analyse.py --split test --final > $L/test_analyse.log 2>&1; say "analyse exit=$?"
$PY scripts/midair_calibrate.py --level mid --base oracle_plane --apply-test --final > $L/test_calibrate.log 2>&1; say "calibrate exit=$?"
$PY scripts/terrain_analyse.py --split test --final > $L/test_terrain_analyse.log 2>&1; say "terrain analyse exit=$?"
say "done"
