# Runbook: finishing the evaluations once data and decisions are in

Commands run from the repository root with the project virtual environment. On Windows, use `.venv/Scripts/python`; elsewhere, `.venv/bin/python`. Test splits run **once**, after every setting is frozen, and the scripts refuse to run them without `--final`.

## Mid-Air (Phase 2)

1. Finish the download. This resumes and skips anything done:
   `python scripts/download_midair.py --only Kite_training/sunny --skip color_left --min-free-gb 10`
2. Check the data: `python scripts/check_data.py`. Then record checksums for the data cards with `python scripts/data_manifest.py`, which is heavy on disk; run it when nothing else is reading the data.
3. Development and calibration evaluation and analysis: `make midair-eval`. Without make, run the five commands under `midair-eval` in the Makefile.
4. Review the development figures in `results/midair/development/figures/`. Revisit the D-209 sampling values only if bins are under-populated, and record any change.
5. Set `ERR_THRESHOLD_FRAC` (D-314), then fit on calibration:
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
4. B3 tiled inference, the tracker grid, and the pixel-noise fit (`SIGMA_PIX_MODEL`) are all on val.
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
- Results schemas: `python scripts/results_schema.py` rewrites `results/<folder>/schema.json` from that folder's files (D-759); `--check` reports a change without writing. Run it only when a new kind of result is added on purpose; `tests/test_results_schema.py` checks every file.
- Showcase clips (D-618) and README media (D-619): `python scripts/auair_export_flight.py --clip 20` and `--clip 44`, then the accurate pipeline run of each (ONNX 960 detector, detection every frame), then `python scripts/make_showcase.py` and `python scripts/make_readme_media.py`. Needs `ffmpeg` on the PATH.

## SynDrone data and truth (Phase 7, D-701 to D-716)

1. Town split, once: `python scripts/make_syndrone_split.py` (whole towns, stratified; refuses to change an existing split).
2. Fetch a town's members: `python scripts/syndrone_fetch.py --town <Town> --modality camera bboxes` (and `rgb` for frames; resumable, checks free disk space against `configs/data.yaml`).
3. Truth, checks and the pipeline on a development town: `bash scripts/queues/syndrone_town.sh <Town>`. The remaining towns: `bash scripts/queues/syndrone_remaining.sh` (test towns get truth only).
4. One sequence by hand: `python scripts/syndrone_run.py`, then `python scripts/syndrone_evaluate.py` (outputs in `cache/syndrone/runs/<name>/`).

## Version 2 detector and tracker (Phase 8, D-715 to D-733)

1. GPU package: `python scripts/visdrone_coco_package.py`; train on Kaggle (docs/gpu_training.md); unzip each run into `cache/train/`.
2. Score each candidate on VisDrone validation: `bash scripts/queues/score_candidate.sh <run> <input size> <rfdetr|dfine>`; compare with `python scripts/candidates.py`.
3. One-time test-dev run, after freezing: `python scripts/cache_detections.py --split test-dev --final ...`, then `python scripts/track_test.py --dets dfine_n_640 --class-map visdrone10 --tracker configs/trackers/motion_dfine_n_640.yaml --final` (refuses if the result exists).

## Ranging, speed and calibration on SynDrone (Phase 9, D-734 to D-746)

All on development and calibration towns; none reads the test towns.
1. Contact-point correction: `python scripts/syndrone_contact_fit.py --run v2_dfine`.
2. Speed display rule: `python scripts/syndrone_speed_display_fit.py` (reads the run's `speed_matches_development.csv.gz`).
3. Interval calibration: `python scripts/syndrone_interval_calibration.py --run <name>`.
4. Fine-tuned detector, leave one town out: `bash scripts/queues/syndrone_finetuned_dfine.sh` (needs `cache/train/dfine_syndrone_minus_<Town>/`).

## End-to-end SynDrone test (Phase 10, D-749, D-753)

1. Check the chain first: `SPLIT=development FRAMES=10 NAME=zz_p10_check bash scripts/queues/phase10_syndrone_test.sh`.
2. Freeze: `python scripts/freeze.py --phase phase10`.
3. The test, once: `bash scripts/queues/phase10_syndrone_test.sh` (pipeline, calibration, evaluation, breakdowns, search measures, then the three navigation grades).

## Camera, IMU and GPS fusion on Mid-Air (Phase 11, D-741, D-744)

1. Development: `python scripts/msckf_midair.py --split development --name <name>`; GPS gate rule: `python scripts/msckf_gps_gate_check.py`; effect on ranging: `python scripts/msckf_ranging.py --nav <name>`.
2. OpenVINS reference (camera and IMU only): `python scripts/msckf_openvins_compare.py --nav msckf_sig4 --ov-dir <OpenVINS output folder>`.
3. Freeze and test once: `python scripts/freeze.py --phase phase11`; `python scripts/msckf_midair.py --split test --final ...`; then `msckf_ranging.py --split test`.

## Edge, map and ROS 2 (Phase 12, D-723, D-745 to D-750)

1. Runtime images for x86-64 and ARM64: `make containers`; same outputs on both: `make platform-check FRAMES=<folder of JPEG frames>` (`scripts/platform_check.py`).
2. Pi 5 speed, estimated: `python scripts/estimate_pi.py --model <name> weights/<model>.onnx`; measured, on a Pi 5: `make pi-benchmark`.
3. ROS 2 integration test in Docker: `bash scripts/ros2_integration.sh <name> <frames dir> <flight CSV> <origin> [frames]` (inputs from `python scripts/ros2_inputs.py`; docs/ros2.md).
4. Search map: `python scripts/make_search_map.py auair --clip 35 --source cache/runs/v2_clip35` or `python scripts/make_search_map.py syndrone --town Town07 --height 50 --source v2_dfine` (docs/map.md).

## Thermal detection and tracking (Phase 13, D-751, D-754, D-757)

1. Package HIT-UAV for the GPU: `python scripts/hituav_coco_package.py` (no test labels packaged).
2. Train on Kaggle: `gpu/train_dfine_hituav_kaggle.py` (docs/gpu_training.md). Two sessions: the two day-split runs, then the official-split run. Unzip each `<run>.zip` into `cache/train/`.
3. Export with parity and score validation: `bash scripts/queues/phase13_thermal_prepare.sh`.
4. BIRDSAI: `python scripts/birdsai_prep.py`, then `bash scripts/queues/phase13_birdsai.sh tune` (training videos; `RUNS="..."` picks the detectors).
5. Freeze, then the one-time tests: `python scripts/freeze.py --phase phase13`; `bash scripts/queues/phase13_thermal_test.sh`; `bash scripts/queues/phase13_birdsai.sh test`.

## Multi-drone search simulation (Phase 14, D-752, D-756)

1. Camera model from frozen runs: `python scripts/multidrone_sensor_fit.py --name p10_v2_cal --split test --allow-test` and `--name v2_dfine_cal_rural --split development`.
2. Design stage (development seeds): `python scripts/multidrone_run.py design --sensor results/multi_drone/sensor_p10_v2_cal.json --workers 12`; check: `python scripts/multidrone_design_check.py . 12`.
3. Fill the choices in `configs/multi_drone.yaml`, set `frozen: true`, `python scripts/freeze.py --phase phase14`.
4. Test stage, once: `python scripts/multidrone_run.py test --sensor results/multi_drone/sensor_p10_v2_cal.json --workers 12`, and for the rural camera with `--only lawnmower_n4 chosen_noshare_n4 chosen_share_n4 --sensor results/multi_drone/sensor_v2_dfine_cal_rural.json`.
5. Byte-budget follow-up (development seeds): `python scripts/multidrone_budget_check.py --workers 12`.
6. Version 2 demo video (D-761): `python scripts/make_demo_v2.py` (needs `cache/maps/syndrone_Town07_h50/` from `make_search_map.py` and `ffmpeg`; writes `cache/video/demo_v2.mp4` and `results/video/demo_v2.json`).
Each run of 12 workers needs about 2 GB of memory; on the 15 GB laptop do not overlap it with detection or tracker grids.

## Confirmed finds and compact sharing (Phase 16, D-764)

1. Measurements from frozen runs: `python scripts/multidrone_clutter_fit.py --name p10_v2_cal --split test --allow-test` and `--name v2_dfine_cal_rural --split development` (false-alarm recurrence, two-kind fit, persistence refits); `python scripts/multidrone_bytes.py --workers 5` (where Phase 14's sharing bytes go, development seeds).
2. Design stages (development seeds), each choice filled into `configs/search16.yaml` before the next: `python scripts/multidrone16_run.py design1 --workers 5`, then `design2`, then `design3`.
3. Set `frozen: true` in `configs/search16.yaml`, then `python scripts/freeze.py --phase phase16`.
4. Test stage, once: `python scripts/multidrone16_run.py test --workers 5`.
5. Paired comparisons for H13 to H16: `python scripts/multidrone16_paired.py test` (reads the test's per-seed file; simulates nothing).
6. Look-alike sensitivity check (D-766; development seeds, after the test): `python scripts/multidrone16_sensitivity.py --workers 5` (2,400 searches, about 1 h 35 min with 5 workers).
Five workers stay within the laptop's memory; do not overlap with a Phase 13 queue.
