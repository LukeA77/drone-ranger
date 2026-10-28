# Runbook: finishing the evaluations once data and decisions are in

Commands run from the repository root with the project virtual environment. On Windows, use `.venv/Scripts/python`; elsewhere, `.venv/bin/python`. Test splits run **once**, after every setting is frozen, and the scripts refuse to run them without `--final`.

## Mid-Air (Phase 2)

1. Finish the download. This resumes and skips anything done:
   `python scripts/download_midair.py --only Kite_training/sunny --skip color_left --min-free-gb 10`
2. Check the data: `python scripts/check_data.py`. Then record checksums for the data cards with `python scripts/data_manifest.py`, which is heavy on disk; run it when nothing else is reading the data.
3. Development and calibration evaluation and analysis: `make midair-eval`. Without make, run the five commands under `midair-eval` in the Makefile.
4. Review the development figures in `results/midair/development/figures/`. Revisit the D-209 sampling values only if bins are under-populated, and record any change.
5. Set `{{ERR_THRESHOLD_FRAC}}` (D-314), then fit on calibration:
   `python scripts/midair_calibrate.py --level mid --base oracle_plane`
6. Freeze. Record the fitted values from `results/midair/calibration_fit.json` in `configs/uncertainty.yaml` and `configs/geometry.yaml`, with sources.
7. Test split, once:

   ```
   python scripts/midair_eval.py --split test --final
   python scripts/midair_analyse.py --split test --final
   python scripts/midair_calibrate.py --level mid --base oracle_plane --apply-test --final
   ```

## Terrain-aware ranging (plans D, B, E; D-401 to D-404)

1. Colour images for development and calibration (test only at the end):
   `python scripts/download_midair.py --only Kite_training/sunny/color_left/<trajectory>`
2. Tracking noise, development only: `python scripts/measure_tracking_noise.py`, then
   `python scripts/midair_terrain.py --set-sigma-from-measurement`
3. Terrain estimates: `python scripts/midair_terrain.py --split development` (then `calibration`)
4. Compare flat and terrain on development: `python scripts/terrain_analyse.py --split development`
5. Fit calibration (B) and the flag (E) on the calibration split: `python scripts/terrain_analyse.py --fit-calibration`
6. Test split, once, after freezing: restore the test flights from Drive (`python scripts/archive_midair.py --restore --splits test`), download their colour images, then
   `python scripts/midair_eval.py --split test --final`,
   `python scripts/midair_terrain.py --split test --final`,
   `python scripts/terrain_analyse.py --split test --final`

## VisDrone (Phase 3)

1. Prepare the data: `make visdrone-prep` (the overlap check, then the MOT ground truth and the YOLO set with the leakage exclusion, D-308).
2. B0 is done for YOLO26n at 640 with ByteTrack. Other B0 variants follow the same pattern:
   `python scripts/track_eval.py --run B0_<tag> --weights weights/yolo26n.pt --class-map coco --imgsz 640 --tracker configs/trackers/bytetrack.yaml`
3. B2, fine-tuning. Run on a GPU (`docs/gpu_training.md`):
   `python scripts/train_detector.py --name B2_yolo26n_640 --model yolo26n.pt --imgsz 640`
   Then track with `--weights cache/train/B2_yolo26n_640/weights/best.pt --class-map visdrone10`.
4. B3 tiled inference, the tracker grid, and the pixel-noise fit (`{{SIGMA_PIX_MODEL}}`) are all on val.
5. Test-dev, once: rerun the frozen configurations with `--split test-dev --final`.

## VisDrone tracker tuning and final run (Phase 3, D-319 to D-321, D-605)

1. Cache validation detections once per detector: `python scripts/cache_detections.py --run B2_yolo26n_960 --weights cache/train/B2_yolo26n_960/weights/best.pt --class-map visdrone10 --imgsz 960`
2. Tracker grid on validation: `python scripts/track_grid.py --dets B2_yolo26n_960 --class-map visdrone10 --tracker configs/trackers/botsort_cmc.yaml --workers 3` (three workers fit in 15 GB of memory), then `python scripts/grid_summary.py`.
3. Detector speed comparison: `python scripts/benchmark_pipeline.py --run B2_yolo26n_960 --weights ... --imgsz 960` (quiet machine).
4. Freeze, then test-dev once: `python scripts/freeze.py --phase phase3 --extra cache/train/B2_yolo26n_960/weights/best.pt configs/trackers/botsort_cmc_tuned_B2_yolo26n_960.yaml`, then `scripts/queues/phase3_test.sh B2_yolo26n_960 960`.
5. Analysis of the saved test-dev predictions (no rerun): `python scripts/track_breakdowns.py --run B2_yolo26n_960_tuned_botsort_cmc --split test-dev --final`; failure gallery on validation: `python scripts/track_tuned.py --dets B2_yolo26n_960` then `python scripts/failure_gallery.py --run B2_yolo26n_960_tuned`.

## Mid-Air final run when disk is short (Phase 2, D-606, D-607)

`scripts/queues/phase2_test.sh`: in batches of three test flights, downloads each from the Mid-Air site (`archive_midair.py --prepare-download`, `download_midair.py`), evaluates, runs terrain, and releases the local copies (`archive_midair.py --release`); then `midair_eval.py --from-cache`, `midair_analyse.py`, `midair_calibrate.py --apply-test`, `terrain_analyse.py`, all with `--final`. It checks the Phase 2 freeze first.

## AU-AIR (Phase 4)

1. `python scripts/inspect_auair.py`; `python scripts/auair_frames.py`; `python scripts/contact_sheets.py` (I label `docs/auair_clip_notes.md` by hand); `python scripts/auair_labels.py`.
2. Detector and tracker on all clips: `python scripts/auair_track.py --run B2_yolo26n_960 --weights cache/train/B2_yolo26n_960/weights/best.pt --imgsz 960`.
3. Camera model on calibration recordings: `python scripts/auair_scale.py`.
4. Stationary segments: `python scripts/auair_stationary.py --run B2_yolo26n_960` and `--run truth --noise-from B2_yolo26n_960`.
5. Tilt from motion: `python scripts/auair_tilt.py --workers 4`.
6. Calibration checks: `python scripts/auair_repeatability.py --run B2_yolo26n_960 --split calibration`; `python scripts/auair_repeatability_view.py --source B2_yolo26n_960 --split calibration`.
7. Freeze and held-out once: `python scripts/freeze.py --phase phase4_view`, then `python scripts/auair_repeatability_view.py --source B2_yolo26n_960 --split held_out --final` (and `--source truth`).
8. Moving targets: `python scripts/auair_moving.py --run B2_yolo26n_960 --tune` (calibration), then without `--tune`.

## Speed (Phase 5, D-601 to D-612)

1. ONNX export with parity: `python scripts/export_onnx.py --weights cache/train/B2_yolo26n_960/weights/best.pt --imgsz 960 --name B2_yolo26n` (and `--imgsz 640 --name B2_yolo26n_960w`).
2. INT8: `python scripts/quantize_onnx.py --onnx weights/B2_yolo26n_960_rect.onnx --head-fp32 model.23/`.
3. Accuracy of each option on validation: `scripts/queues/phase5_accuracy.sh`, `scripts/queues/phase5_stride.sh`.
4. Benchmarks on a quiet, plugged-in machine (sleep is blocked by `scripts/queues/keep_awake.py` on Windows): `scripts/queues/phase5_bench.sh`, `scripts/queues/phase5_bench2.sh`.
5. Chart: `python scripts/pareto.py`.

## End-to-end pipeline and documents (Phase 6, D-613)

- Synthetic demo (no datasets): `aerialrange demo --out cache/demo`.
- A real flight: `python scripts/auair_export_flight.py --clip 35`, then `aerialrange run --source cache/auair/flights/clip_35/frames --metadata cache/auair/flights/clip_35/metadata.csv --out cache/runs/auair_clip35`.
- Documents: edit `docs/src/*.md`, then `python scripts/build_docs.py` (rebuilds `results/headline.json`, fills every number, refuses unresolved placeholders and banned phrases).
- Showcase clips (D-618) and README media (D-619): `python scripts/auair_export_flight.py --clip 20` and `--clip 44`, then the accurate pipeline run of each (ONNX 960 detector, detection every frame), then `python scripts/make_showcase.py` and `python scripts/make_readme_media.py`. Needs `ffmpeg` on the PATH.
