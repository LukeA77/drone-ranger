# SynDrone evaluation: detect, track, range and speed against known truth

*Every number here is filled from `results/syndrone/`. Development towns only: these results were used to set thresholds, so they are not test results. The test towns are run once, after the settings are frozen.*

## Why SynDrone

Version 1 scored each stage on a different dataset, because no single dataset had objects, tracks and true distances together (see [limitations](limitations.md)). SynDrone (Rizzoli et al., 2023; Creative Commons Attribution, see the [data card](model_and_data.md)) is rendered in the CARLA simulator: eight towns, each filmed by drone cameras at 20, 50 and 80 m, with every car and person's 3D box, the camera pose, depth and semantic labels in every frame. That makes it possible to run the whole pipeline (detection, tracking, ranging and speed) and score every output against known truth.

It is synthetic: textures, lighting and people's appearance differ from real footage, and the camera pose is exact. Results here measure the pipeline's geometry and its detector's transfer to rendered scenes, not real-world accuracy.

## Data and split

Towns, not frames, are split (D-701), so no scene appears in both development and test. One town from each of three kinds of town is used for development, one for test, and the remaining two for calibration:

| Split | Towns |
|---|---|
| Development | {{RESULT:splits.json#syndrone.splits.development.0}}, {{RESULT:splits.json#syndrone.splits.development.1}}, {{RESULT:splits.json#syndrone.splits.development.2}} |
| Calibration | {{RESULT:splits.json#syndrone.splits.calibration.0}}, {{RESULT:splits.json#syndrone.splits.calibration.1}} |
| Test | {{RESULT:splits.json#syndrone.splits.test.0}}, {{RESULT:splits.json#syndrone.splits.test.1}}, {{RESULT:splits.json#syndrone.splits.test.2}} |

Each development town has three sequences (one per camera height) of 3,000 frames at 25 frames per second. Frames are read from the published archives on demand and checked against each archive's CRC-32 checksums (D-702, D-708).

## How the truth is built

- **Camera height.** SynDrone stores each camera's height above an unpublished base height that changes from frame to frame. It is recovered per frame by fitting the stored 3D boxes to the depth images and instance masks (D-704). The three cameras, fitted independently, agree to within a few centimetres; the median standard error of the fit is {{RESULT:syndrone/development/z0_check_Town07.json#z0_se_m.median|.3f}} m on {{RESULT:syndrone/development/z0_check_Town07.json#town}}.
- **Boxes.** Projected 3D boxes match the instance-mask boxes to a median of {{RESULT:syndrone/development/z0_check_Town01.json#reprojection.h20.median_abs_edge_px|.0f}} to {{RESULT:syndrone/development/z0_check_Town07.json#reprojection.h80.median_abs_edge_px|.0f}} pixels per edge (D-706). The scoring uses the instance-mask box, which is what a detector can see.
- **Which objects count.** People are walkers; vehicles are cars, trucks and buses. An object counts if at least {{RESULT:syndrone/development/eval_v1_accurate.json#truth_rules.min_visibility|.0%}} of it is visible and it covers at least {{RESULT:syndrone/development/eval_v1_accurate.json#truth_rules.min_instance_px}} pixels (D-705). Objects below that, riders, two-wheelers, trams, static map props and walkers stacked on one spot by a simulator glitch (D-712) are ignored: finding them is neither rewarded nor penalised, as in the MOTChallenge evaluation tools.
- **Range** is the distance from the camera to the object's footprint centre on the ground, the point the pipeline ranges to.
- **Speed** is the ground speed from the object's 3D position over a three-frame window; position noise in the simulator contributes about {{RESULT:syndrone/development/speed_truth_Town01.json#speed_sigma_from_jitter_ms|.3f}} m/s (D-704).

Every metric is computed per sequence and pooled; 95 % intervals resample whole sequences (9 units) and, separately, whole towns (3 units, a coarse check on how much the result depends on which towns were drawn).

## Results: version 1 detector, no SynDrone training

The version 1 detector (YOLO26n at 960 pixels, trained on VisDrone) and the version 1 tracker, run at full resolution on every frame (`configs/run_accurate.yaml`).

### Detection and tracking

| | People | Vehicles |
|---|---|---|
| Valid truth views | {{RESULT:syndrone/development/eval_v1_accurate.json#person.n_valid_truth|,}} | {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.n_valid_truth|,}} |
| Detection AP50 | {{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.ap50_all|.3f}} ({{RESULT:syndrone/development/eval_v1_accurate.json#person.ci95_resample_sequences.ap50_all.0|.3f}} to {{RESULT:syndrone/development/eval_v1_accurate.json#person.ci95_resample_sequences.ap50_all.1|.3f}}) | {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.ap50_all|.3f}} ({{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.ci95_resample_sequences.ap50_all.0|.3f}} to {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.ci95_resample_sequences.ap50_all.1|.3f}}) |
| AP50 on VisDrone MOT validation, for comparison | {{RESULT:headline.json#model.mot_val_person_ap50|.3f}} | {{RESULT:headline.json#model.mot_val_vehicle_ap50|.3f}} |
| IDF1 | {{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.idf1|.3f}} | {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.idf1|.3f}} ({{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.ci95_resample_sequences.idf1.0|.3f}} to {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.ci95_resample_sequences.idf1.1|.3f}}) |
| HOTA | {{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.hota_HOTA|.3f}} | {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.hota_HOTA|.3f}} |
| MOTA | {{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.mota|.3f}} | {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.mota|.3f}} |

Intervals in brackets resample sequences.

- **Vehicles transfer; people do not.** Vehicle AP50 is close to VisDrone's. People are almost never found: AP50 is {{RESULT:syndrone/development/eval_v1_accurate.json#person.by_height.h20.ap50_all|.3f}} from 20 m and {{RESULT:syndrone/development/eval_v1_accurate.json#person.by_height.h50.ap50_all|.3f}} from 50 m. Most rendered people are small (AP50 for people under 16 pixels tall is {{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.ap50_0-16|.3f}}), and they may also look unlike VisDrone's people. Fine-tuning on SynDrone (below) measures how much of this is appearance and how much is size.
- **Size matters for vehicles too.** Vehicle AP50 is {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.ap50_0-16|.3f}} under 16 pixels, {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.ap50_16-32|.3f}} at 16 to 32 and {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.ap50_32-64|.3f}} at 32 to 64.
- **Tracking is weaker than on VisDrone** (vehicle IDF1 {{RESULT:headline.json#visdrone_test.vehicle_idf1|.3f}} on the VisDrone test set). The association component of HOTA is {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.hota_AssA|.3f}} against {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.hota_DetA|.3f}} for detection, so most of the loss is identity, not detection.

### Range

Over {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.range_n|,}} matched vehicle track-frames:

- The median range error is {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.range_rel_err_median_abs|.1%}}, and one in ten is more than {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.range_rel_err_p90_abs|.1%}} off. This uses SynDrone's exact camera pose, so it is the geometry's error alone, not a field result.
- Ranges are slightly short from low cameras: median signed error {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.by_height.h20.range_rel_err_median_signed|.1%}} at 20 m, {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.by_height.h50.range_rel_err_median_signed|.1%}} at 50 m and {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.by_height.h80.range_rel_err_median_signed|.1%}} at 80 m. The likely cause is the point the box is ranged from, which is not exactly the footprint centre; its correction is measured separately.
- **Error bars are too wide.** {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.range_coverage_0.90|.0%}} of errors fall inside the 90 % interval (and {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.range_coverage_0.68|.0%}} inside the 68 % one). The range uncertainty comes from how much ranges to parked cars wobble in real AU-AIR footage (D-517), which is several times what a simulator with exact poses produces.

People are ranged too ({{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.range_n|,}} matched track-frames, median error {{RESULT:syndrone/development/eval_v1_accurate.json#person.pooled.range_rel_err_median_abs|.1%}}), but almost all come from the 20 m camera, where they are found at all.

### Speed

- Over {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.speed_n|,}} vehicle track-frames, the median speed error is {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.speed_err_median_abs_ms|.2f}} m/s, and one in ten is more than {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.speed_err_p90_abs_ms|.1f}} m/s off. {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.speed_coverage_0.90|.0%}} of errors fall inside the 90 % interval: speed error bars are slightly too narrow, the reverse of range.
- **Speed is not shown yet (D-710).** The overlay shows a speed only when the filter is confident enough that 9 in 10 shown speeds are within {{RESULT:syndrone/development/speed_display_fit_v1_accurate.json#tolerance_ms|.1f}} m/s, the accuracy needed to tell a walking person from a standing one. No setting of the version 1 filter reaches that: the most accurate setting that shows at least 100 speeds is right {{RESULT:syndrone/development/speed_display_fit_v1_accurate.json#most_accurate.accuracy|.0%}} of the time. So no speed is shown ("speed: settling"), and the rule is refitted once the filter's noise is calibrated.

## Results: after SynDrone fine-tuning

The version 1 detector was fine-tuned on SynDrone in three folds, leaving one development town out each time, and each town is scored with the model that never saw it (D-707, [GPU training](gpu_training.md)). Both columns use the project's own tracker (D-713), so the only difference is the detector. Swapping version 1's tracker for the project's own changes little on SynDrone: vehicle IDF1 goes from {{RESULT:syndrone/development/eval_v1_accurate.json#vehicle.pooled.idf1|.3f}} to {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.idf1|.3f}} with the version 1 detector.

| | People, no SynDrone training | People, fine-tuned | Vehicles, no SynDrone training | Vehicles, fine-tuned |
|---|---|---|---|---|
| Detection AP50 | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.ap50_all|.3f}} ({{RESULT:syndrone/development/eval_v2trk_accurate.json#person.ci95_resample_sequences.ap50_all.0|.3f}} to {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.ci95_resample_sequences.ap50_all.1|.3f}}) | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.ap50_all|.3f}} ({{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.ci95_resample_sequences.ap50_all.0|.3f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.ci95_resample_sequences.ap50_all.1|.3f}}) | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.ap50_all|.3f}} ({{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.ci95_resample_sequences.ap50_all.0|.3f}} to {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.ci95_resample_sequences.ap50_all.1|.3f}}) | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.ap50_all|.3f}} ({{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.ci95_resample_sequences.ap50_all.0|.3f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.ci95_resample_sequences.ap50_all.1|.3f}}) |
| AP50, under 16 px | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.ap50_0-16|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.ap50_0-16|.3f}} | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.ap50_0-16|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.ap50_0-16|.3f}} |
| AP50, 16 to 32 px | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.ap50_16-32|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.ap50_16-32|.3f}} | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.ap50_16-32|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.ap50_16-32|.3f}} |
| AP50, 32 to 64 px | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.ap50_32-64|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.ap50_32-64|.3f}} | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.ap50_32-64|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.ap50_32-64|.3f}} |
| IDF1 | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.idf1|.3f}} ({{RESULT:syndrone/development/eval_v2trk_accurate.json#person.ci95_resample_sequences.idf1.0|.3f}} to {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.ci95_resample_sequences.idf1.1|.3f}}) | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.idf1|.3f}} ({{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.ci95_resample_sequences.idf1.0|.3f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.ci95_resample_sequences.idf1.1|.3f}}) | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.idf1|.3f}} ({{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.ci95_resample_sequences.idf1.0|.3f}} to {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.ci95_resample_sequences.idf1.1|.3f}}) | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.idf1|.3f}} ({{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.ci95_resample_sequences.idf1.0|.3f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.ci95_resample_sequences.idf1.1|.3f}}) |
| HOTA | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.hota_HOTA|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.hota_HOTA|.3f}} | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.hota_HOTA|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.hota_HOTA|.3f}} |
| MOTA | {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.mota|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.mota|.3f}} | {{RESULT:syndrone/development/eval_v2trk_accurate.json#vehicle.pooled.mota|.3f}} | {{RESULT:syndrone/development/eval_ft_loto_accurate.json#vehicle.pooled.mota|.3f}} |

Intervals in brackets resample sequences.

- **People: both appearance and size.** Fine-tuning raises person AP50 in all {{RESULT:syndrone/development/compare_v2trk_accurate_vs_ft_loto_accurate.json#person.ap50_all.n_sequences|d}} sequences, about five-fold overall, so much of the zero-shot failure was appearance: rendered people look unlike VisDrone's. Size is the rest: people 32 to 64 pixels tall are now found well (AP50 {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.ap50_32-64|.3f}}), people under 16 pixels still are not ({{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.ap50_0-16|.3f}}). From 80 m, person AP50 rises from {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.by_height.h80.ap50_all|.3f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.by_height.h80.ap50_all|.3f}}.
- **Vehicles: better in two towns, worse in the third.** Vehicle AP50 improves in {{RESULT:syndrone/development/compare_v2trk_accurate_vs_ft_loto_accurate.json#vehicle.ap50_all.n_improved|d}} of {{RESULT:syndrone/development/compare_v2trk_accurate_vs_ft_loto_accurate.json#vehicle.ap50_all.n_sequences|d}} sequences. By town, the mean change is {{RESULT:syndrone/development/compare_v2trk_accurate_vs_ft_loto_accurate.json#vehicle.ap50_all.town_mean.Town01|+.3f}} on Town01, {{RESULT:syndrone/development/compare_v2trk_accurate_vs_ft_loto_accurate.json#vehicle.ap50_all.town_mean.Town07|+.3f}} on Town07 and {{RESULT:syndrone/development/compare_v2trk_accurate_vs_ft_loto_accurate.json#vehicle.ap50_all.town_mean.Town10HD|+.3f}} on Town10HD, a dense city centre, whose model trained on a small town (Town01) and a rural one (Town07). Training on two rendered towns can cost accuracy on a third, so the gain is town-dependent, and the town-level intervals (three towns) are wide.
- **Tracking follows detection.** Person IDF1 rises from {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.idf1|.3f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.idf1|.3f}}; more people found also means more identity switches ({{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.id_switches|,.0f}} to {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.id_switches|,.0f}}).
- **Range and speed hold up on far more objects.** With the fine-tuned detector, {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.range_n|,}} person track-frames are ranged (from {{RESULT:syndrone/development/eval_v2trk_accurate.json#person.pooled.range_n|,}}), at a median error of {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.range_rel_err_median_abs|.1%}}; person speed error is {{RESULT:syndrone/development/eval_ft_loto_accurate.json#person.pooled.speed_err_median_abs_ms|.2f}} m/s median.

These are development results: the fine-tuning recipe was fixed before scoring, but the same towns are used throughout this phase. The fine-tuned detector is not used anywhere else: it is a measurement of the gap between VisDrone and SynDrone, and version 2's detector is chosen separately (D-715).

## Limits of this evaluation

- Synthetic scenes and exact camera poses: real-flight errors will be larger.
- Three development towns. The town-level intervals rest on three units and are a coarse guide only.
- The same towns set the truth rules (visibility, reprojection, walker stacks) and the speed display rule, so these are development results; the test towns give the unbiased figures.
- The detector and filter are version 1's (the fine-tuned detector apart), so the numbers are a baseline for the version 2 changes.
