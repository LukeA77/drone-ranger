<div align="center">

# Aerial Track and Range

**Detect and track people and vehicles from a single drone camera, estimate how far away each one is with an error bar, and run on a laptop CPU.**

**[Luke Andrew](https://lukeandrew.dev)**

![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab)
![Licence AGPL-3.0](https://img.shields.io/badge/licence-AGPL--3.0--only-2a78d6)
![Runs on CPU](https://img.shields.io/badge/runs%20on-laptop%20CPU-1baf7a)

<img src="docs/media/tracking_hero.webp" width="900" alt="Busy city crossing seen from a drone: every car and pedestrian has a coloured box with an identity number that stays with it as it moves">

<sub>VisDrone test-dev, the held-out test set, never used for training or tuning: the fine-tuned detector and tuned tracker, each colour and number one identity kept across frames. Clips chosen for clarity; across the whole test set, identity F1 is {{RESULT:headline.json#visdrone_test.vehicle_idf1|.3f}} for vehicles and {{RESULT:headline.json#visdrone_test.person_idf1|.3f}} for people.</sub>

</div>

## Results at a glance

Every number on this page is generated from a results file in [`results/`](results/) by code in this repository. Each test split was run once, after all settings were frozen.

| | Result | Data |
|---|---|---|
| **Distance accuracy** | Median error **{{RESULT:headline.json#midair_test.paired_terrain_median|.1%}}** (95 % interval {{RESULT:headline.json#midair_test.paired_terrain_median_ci.0|.1%}} to {{RESULT:headline.json#midair_test.paired_terrain_median_ci.1|.1%}}) against {{RESULT:headline.json#midair_test.paired_flat_median|.1%}} for flat ground given the true height, on the same samples; 90th percentile {{RESULT:headline.json#midair_test.paired_terrain_p90|.0%}} against {{RESULT:headline.json#midair_test.paired_flat_p90|.0%}} | Mid-Air, {{RESULT:headline.json#midair_test.n_test_flights|d}} synthetic test flights with exact truth; the {{RESULT:headline.json#midair_test.paired_share|.0%}} of samples where both methods give an answer |
| **With realistic errors** | With assumed typical navigation errors the median rises to **{{RESULT:headline.json#robustness.nav_typical_terrain|.1%}}** (flat ground: {{RESULT:headline.json#robustness.nav_typical_flat|.0%}}); ranging a detected car's box rather than a ground point adds a {{RESULT:headline.json#robustness.car_contact_median_abs|.0%}} short bias, {{RESULT:headline.json#robustness.car_end_to_end_median|.1%}} end to end even with exact poses | Mid-Air, {{RESULT:headline.json#robustness.n_dev_flights|d}} development flights; simulated boxes |
| **Error bars** | After calibration, nominal 90 % intervals contain the truth **{{RESULT:headline.json#midair_test.cov_cal_90|.0%}}** of the time ({{RESULT:headline.json#midair_test.cov_raw_90|.0%}} before): short of 90 %, a sign that flights differ more than calibration assumes | Mid-Air test |
| **Tracking** | Identity F1 **{{RESULT:headline.json#visdrone_test.vehicle_idf1|.3f}}** for vehicles, {{RESULT:headline.json#visdrone_test.person_idf1|.3f}} for people; only {{RESULT:headline.json#visdrone_test.person_recall_16_32|.0%}} of people 16 to 32 pixels tall are tracked | VisDrone test-dev; a standard detector and tracker, fine-tuned |
| **Real footage** | Parked vehicles' positions repeat to **{{RESULT:headline.json#auair.heldout_rel_all|.1%}}** of their distance (repeatability, not accuracy); the detector reaches AP50 only **{{RESULT:headline.json#auair.heldout_vehicle_ap50|.2f}}** for vehicles and {{RESULT:headline.json#auair.heldout_person_ap50|.2f}} for people here | AU-AIR, held-out flights |
| **Speed** | **{{RESULT:headline.json#edge.accurate_fps|.1f}} frames per second** at full accuracy; {{RESULT:headline.json#edge.fast_fps|.1f}} with detection on every second frame. Mean IDF1 on validation {{RESULT:headline.json#edge.accurate_idf1|.3f}} and {{RESULT:headline.json#edge.fast_idf1|.3f}} (not comparable with the test-set tracking row) | {{RESULT:headline.json#edge.cpu|s}}, no GPU |

## How it works

```mermaid
flowchart LR
    A[Video frame] --> B[Detector<br/>YOLO26n, fine-tuned on drone imagery]
    B --> C[Tracker<br/>BoT-SORT with camera-motion compensation]
    C --> D[Ranging<br/>ray from the camera to the ground]
    G[Drone pose and height] --> D
    H[Terrain from motion<br/>local ground surface] --> D
    D --> E[Uncertainty<br/>propagated, calibrated, flagged]
    E --> F[Kalman filter<br/>position, speed, heading]
    F --> O[tracks.csv and annotated video]
```

A camera gives direction, not distance. Each tracked box's ground-contact point is turned into a ray and intersected with the ground: either a flat plane, or a local surface triangulated from image features tracked across frames using the drone's known motion ("terrain from motion"), which needs no height above the ground. Sensor noise is propagated to every range, the intervals are calibrated on separate flights (split conformal prediction), and a small, interpretable model flags estimates that are likely to be badly wrong.

## 1. Detect and track

<p align="center">
  <img src="docs/media/tracking.webp" width="860" alt="Road junction at night seen from a drone: cars and pedestrians each carry a coloured box with an identity number">
</p>

A YOLO26n detector fine-tuned on VisDrone, with the BoT-SORT tracker and camera-motion compensation, its thresholds tuned on validation data. Each colour and number is one identity kept across frames. The clip above is from test-dev (night, held out), replayed from the saved one-time test run.

<p align="center">
  <img src="docs/media/before_after.webp" width="860" alt="Side by side on the same top-down crossroads: on the left an off-the-shelf detector boxes almost nothing; on the right the fine-tuned detector boxes nearly every vehicle">
</p>

**Left:** the off-the-shelf detector. **Right:** after fine-tuning and tuning. This comparison is on a validation clip, the data the thresholds were tuned on, because the off-the-shelf baseline was never run on test-dev. Mean identity F1 on validation rises from {{RESULT:headline.json#visdrone_val.b0_cmc_mean_idf1|.2f}} to {{RESULT:headline.json#visdrone_val.b2_960_tuned|.2f}}. On the test set, vehicles reach IDF1 {{RESULT:headline.json#visdrone_test.vehicle_idf1|.3f}} (MOTA {{RESULT:headline.json#visdrone_test.vehicle_mota|.3f}}) and people {{RESULT:headline.json#visdrone_test.person_idf1|.3f}}. This is a standard detector-and-tracker design, and most of the gain comes from fine-tuning on drone imagery; it is not compared with published VisDrone tracking results, whose protocol (classes and test split) differs. Small people are the weak point: {{RESULT:headline.json#visdrone_test.person_recall_16_32|.0%}} are found at 16 to 32 pixels tall, against {{RESULT:headline.json#visdrone_test.person_recall_128|.0%}} above 128.

## 2. Range with honest uncertainty

<table>
<tr>
<td width="42%"><img src="docs/media/range_flag.webp" alt="Synthetic flight over grass and rocks: dots on the ground mark range estimates; circles were within 10 % of the truth, crosses were further off; amber marks where the reliability flag was raised"></td>
<td><img src="results/midair/test/terrain/figures/terrain_vs_flat_slope_deg.png" alt="Chart: median range error against ground slope for three methods; terrain from motion stays lowest at every slope"></td>
</tr>
</table>

On synthetic flights, where the true distance of every pixel is known, each dot is a range estimate that stays on its spot on the ground: **circles** were within 10 % of the truth and **crosses** were not; **amber** means the reliability flag was raised. Compared on the same samples with flat ground given the true height (a height a real drone does not have), terrain from motion has a median error of {{RESULT:headline.json#midair_test.paired_terrain_median|.1%}} against {{RESULT:headline.json#midair_test.paired_flat_median|.1%}} and is closer on {{RESULT:headline.json#midair_test.paired_terrain_closer|.0%}} of samples; it has the lower median on {{RESULT:headline.json#midair_test.paired_flights_better|d}} of {{RESULT:headline.json#midair_test.paired_n_flights|d}} flights, but not on flight {{RESULT:headline.json#midair_test.paired_worst_flight|d}} ({{RESULT:headline.json#midair_test.paired_worst_flight_terrain|.1%}} against {{RESULT:headline.json#midair_test.paired_worst_flight_flat|.1%}}), and one estimate in ten is still more than {{RESULT:headline.json#midair_test.paired_terrain_p90|.0%}} off. Its error stays at or below {{RESULT:headline.json#midair_test.terrain_err_slope_30_45|.1%}} on slopes up to 45°; a flat-ground model, even given the true height, grows from {{RESULT:headline.json#midair_test.flat_err_slope_0_5|.0%}} on level ground to {{RESULT:headline.json#midair_test.flat_err_slope_45_90|.0%}} on slopes above 45°. After calibration, 90 % error bars hold only {{RESULT:headline.json#midair_test.cov_cal_90|.0%}} of the time: calibration assumes the test flights resemble the calibration flights, and the shortfall shows they do not quite; the flag (AUROC {{RESULT:headline.json#midair_test.flag_auroc|.2f}}) catches {{RESULT:headline.json#midair_test.flag_recall|.0%}} of estimates more than 10 % off.

### How far the number transfers

The figures above give the method the simulator's exact camera poses and an ideal camera. A real drone's navigation is not exact, so every pose the method uses was perturbed and the development flights re-run ([pose sweep](scripts/terrain_pose_sweep.py)). On these development flights, exact poses give {{RESULT:headline.json#robustness.exact_terrain|.1%}} for terrain from motion and {{RESULT:headline.json#robustness.exact_flat|.1%}} for flat ground given the true height (the test-flight figures above are on different flights and samples).

<p align="center">
  <img src="results/midair/development/terrain/figures/pose_sweep_pos_jitter.png" width="640" alt="Chart: median range error against per-frame position noise; terrain from motion rises from about 6 % at 2 cm to about 66 % at 1 m, crossing above the flat-ground line near 7 cm">
</p>

- **Errors shared by every frame do not matter much.** A 5 m GPS offset leaves terrain from motion at {{RESULT:headline.json#robustness.pos_bias_5m_terrain|.1%}} (flat ground: {{RESULT:headline.json#robustness.pos_bias_5m_flat|.0%}}), and a 2° tilt error raises it only to {{RESULT:headline.json#robustness.att_bias_2deg_terrain|.1%}}, because the camera and the triangulated ground move together.
- **Errors that change from frame to frame matter a lot.** Triangulation uses frames 0.04 to 0.4 s apart, so it needs the change in pose between them to be accurate: per-frame position noise of 2 cm gives {{RESULT:headline.json#robustness.pos_jitter_0p02m_terrain|.1%}}, 10 cm gives {{RESULT:headline.json#robustness.pos_jitter_0p1m_terrain|.1%}} (worse than flat ground with the true height, {{RESULT:headline.json#robustness.pos_jitter_0p1m_flat|.0%}}), and attitude noise of 0.1° gives {{RESULT:headline.json#robustness.att_jitter_0p1deg_terrain|.1%}}. In practice this calls for the short-term accuracy of an inertial or visual-inertial navigation system, not raw GNSS.
- **Three assumed navigation grades** (combining both kinds of error; assumptions, not product specifications) give {{RESULT:headline.json#robustness.nav_good_terrain|.1%}}, {{RESULT:headline.json#robustness.nav_typical_terrain|.1%}} and {{RESULT:headline.json#robustness.nav_poor_terrain|.0%}}, against {{RESULT:headline.json#robustness.nav_good_flat|.0%}}, {{RESULT:headline.json#robustness.nav_typical_flat|.0%}} and {{RESULT:headline.json#robustness.nav_poor_flat|.0%}} for flat ground given the true height.
- **The box adds its own error.** The pipeline ranges a detected box's bottom-centre, which for a car seen at an angle is the near edge, not the centre. Simulated cars with the detector's measured box noise read {{RESULT:headline.json#robustness.car_contact_median_abs|.1%}} short (people {{RESULT:headline.json#robustness.person_contact_median_abs|.1%}}); combined with the ranging error, a car is {{RESULT:headline.json#robustness.car_end_to_end_median|.1%}} off at the median and more than {{RESULT:headline.json#robustness.car_end_to_end_p90|.0%}} off one time in ten, even with exact poses.

Lens distortion, rolling shutter and focal-length error are not modelled: Mid-Air renders an ideal camera.

## 3. Real footage

<table>
<tr>
<td width="50%"><img src="docs/media/ranging_real_late.webp" alt="Oblique view of a roundabout with vehicles labelled with distance and error bar"></td>
<td width="50%"><img src="docs/media/ranging_overhead.webp" alt="Straight-down view of a roundabout with two cars labelled 32 ± 1 m and 37 ± 3 m"></td>
</tr>
</table>

AU-AIR records real flights with GPS and altitude but no camera calibration and no camera tilt, so both were estimated: the focal length from GPS motion against image motion on separate calibration flights ({{RESULT:headline.json#auair.focal_px|.0f}} px), and the tilt, frame by frame, from the motion of the ground in the footage itself ([camera model](docs/auair_camera_model.md)). On held-out flights, parked vehicles' estimated positions repeat to {{RESULT:headline.json#auair.heldout_rel_all|.1%}} of their distance ({{RESULT:headline.json#auair.heldout_segments|d}} vehicle segments). There is no ground truth here, so this is repeatability, not accuracy. The detector itself transfers poorly to this footage: AP50 {{RESULT:headline.json#auair.heldout_vehicle_ap50|.2f}} for vehicles and {{RESULT:headline.json#auair.heldout_person_ap50|.2f}} for people on the held-out flights, partly because AU-AIR's boxes are drawn looser than VisDrone's (D-512); the rest is a genuine gap between the two datasets, and many vehicles are missed. Two further warning signs: the tilt is estimated from the same footage it is used on, and {{RESULT:headline.json#auair.vehicle_share_above_70|.0%}} of vehicle tracks imply speeds above 70 km/h at a junction where that is implausible, most likely from shallow lines of sight and identity switches.

## 4. Speed on a laptop CPU

<p align="center">
  <img src="results/edge/figures/pareto.png" width="720" alt="Chart: tracking accuracy against time per frame for every measured setting, with the best trade-offs joined and the 30 fps and 10 fps budgets marked">
</p>

Every setting was measured on the same laptop CPU ({{RESULT:headline.json#edge.cpu|s}}). At full accuracy (ONNX, 960-pixel input, detection on every frame) the pipeline runs at **{{RESULT:headline.json#edge.accurate_fps|.1f}} frames per second** (mean IDF1 on validation {{RESULT:headline.json#edge.accurate_idf1|.3f}}). To reach video rate, a 640-pixel input, detection on every second frame (the tracker predicts in between) and detection overlapped with tracking give {{RESULT:headline.json#edge.fast_fps|.1f}} frames per second, at the cost of mean validation IDF1 {{RESULT:headline.json#edge.fast_idf1|.3f}}. Each frame's result then arrives {{RESULT:headline.json#edge.fast_latency_median_ms|.0f}} ms after capture (median), so 30 fps is met in throughput, not in per-frame delay.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt     # .venv/bin/python on Linux or macOS
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest                                   # run the test suite
.venv/Scripts/aerialrange demo --out cache/demo                  # full pipeline on a synthetic flight, no datasets needed
```

The demo writes `tracks.csv`, `annotated.mp4` and `timing.json`, and checks every distance and speed against the synthetic truth. To run on a real flight (frames or a video, plus a flight-data CSV with position, height, heading and camera tilt per frame):

```bash
.venv/Scripts/aerialrange run --source <frames-or-video> --metadata <flight.csv> --out cache/run
```

## Documentation

| Document | What it covers |
|---|---|
| [Technical brief](docs/brief.md) | Two-page summary: problem, method, results, failure modes |
| [Limitations, and questions and answers](docs/limitations.md) | Where it breaks, by how much, and the questions an engineer is likely to ask |
| [Model and data](docs/model_and_data.md) | Detector, tracker, datasets and licences |
| [AU-AIR camera model](docs/auair_camera_model.md) | Focal length, tilt and altitude inferred from footage |
| [Decision log](docs/decisions.md) | Every design choice, with its reason and evidence |
| [Runbook](docs/runbook.md) | The commands that reproduce each result, in order |

## Repository layout

| Path | Contents |
|---|---|
| `src/aerialrange/geometry/` | Camera model, ranging, uncertainty propagation, triangulation, terrain fitting, Kalman filter (built test-first) |
| `src/aerialrange/track/`, `detect/` | Tracking, evaluation, tiling, saved-detection replay |
| `src/aerialrange/evaluate/` | Mid-Air and AU-AIR evaluations, conformal calibration, reliability flag, statistics |
| `src/aerialrange/pipeline/` | The end-to-end `aerialrange run` and `demo` |
| `src/aerialrange/edge/` | Latency benchmark harness |
| `scripts/` | One script per experiment |
| `configs/` | Every setting, each with its source |
| `results/` | Every number and figure quoted in the documents |


## The main gap

No single test runs detect, track and range end to end against known object distances. Tracking is scored on VisDrone (boxes, no distances), ranging on Mid-Air (distances, no objects; exact poses) and real footage on AU-AIR (neither true distances nor a well-transferred detector). The box-to-ground simulation and the pose-noise sweep narrow the gap but are still separate pieces. The most valuable next step is one setup where every object's distance is known: a drone flown past a car carrying a GPS logger, or a simulator with vehicles and people (for example CARLA), scored through the whole pipeline.

## Scope

A perception and measurement project on public research datasets. People and vehicles are tracked as boxes only: no face analysis and no re-identification by appearance. Distances hold under stated assumptions (a known camera and pose, and flat ground or terrain from motion) and are not claimed beyond the tested conditions: low altitude ({{RESULT:headline.json#auair.alt_min_m|.0f}} to {{RESULT:headline.json#auair.alt_max_m|.0f}} m on real footage), short ranges, and one site for real footage.

## Datasets and credits

No datasets or model weights are stored in this repository; the short clips in `docs/media/` contain frames from the datasets below and are shared under their terms.

- **VisDrone2019**: P. Zhu, L. Wen, D. Du et al., "Detection and tracking meet drones challenge", *IEEE TPAMI*, 2021. Research use.
- **Mid-Air**: M. Fonder and M. Van Droogenbroeck, "Mid-Air: a multi-modal dataset for extremely low altitude drone flights", *CVPR Workshops*, 2019. CC BY-NC-SA 4.0.
- **AU-AIR**: I. Bozcan and E. Kayacan, "AU-AIR: a multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance", *ICRA*, 2020. CC BY 4.0.
- Detector and trackers: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

## Licence

[GNU Affero General Public License v3.0 only](LICENSE). The detector and trackers come from Ultralytics, which is AGPL-3.0, so code combined with it must be released under the same terms; in practice, it cannot go into a closed product as it stands. Only two modules use Ultralytics (`src/aerialrange/track/runner.py` and `src/aerialrange/pipeline/run.py`). The ranging, uncertainty and evaluation code (`src/aerialrange/geometry/`, `src/aerialrange/evaluate/`) does not, and I am happy to discuss a permissive licence for those parts.

## Author

**Luke Andrew**, [lukeandrew.dev](https://lukeandrew.dev), [GitHub](https://github.com/LukeA77).
