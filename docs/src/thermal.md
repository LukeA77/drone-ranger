# Thermal detection (Phase 13)

People are warm against cool ground, so a thermal camera can show a person a colour camera misses, at night and in poor light. This phase trains the version 2 detector's architecture (D-FINE N at 640) on thermal drone images, scores it on days it never saw, and measures how well detectors carry over between colour and thermal. Design: D-751 in [decisions.md](decisions.md).

## Data

HIT-UAV (Suo et al., *Scientific Data* 10, 227, 2023; CC BY 4.0, D-748): 2,898 thermal images, 640 x 512, from a DJI Zenmuse XT2 (FLIR longwave sensor, 25 mm lens) on a DJI Matrice M210 V2, flown at 60 to 130 m with the camera 30 to 90 degrees down, by day and night, over schools, car parks, roads and playgrounds. Each file name records the time of day, altitude and camera angle, so results can be broken down by all three.

Classes are mapped to the version 2 detector's own (VisDrone numbers), so its output and evaluation groups are unchanged: Person to pedestrian, Car to car, Bicycle to bicycle (in neither group). DontCare (objects the annotators could not identify, possibly people) and OtherVehicle (not defined further) are ignored regions: detections on them neither count for nor against.

## Split

The dataset's own split divides images 70 / 10 / 20 at random. But the images are frames from videos, and most test images sit one to four frames from a training image of the same scene: a test there measures recognising scenes already seen. So the main split keeps whole capture days apart (9 days; validation 23 January 2021, test 16 and 21 January 2021, together 23 % of images). The test days cover every altitude, day and night. The official split is still used once, to compare with published results and to show what the shared videos add (`results/hituav/splits.json`, `scripts/hituav_coco_package.py`).

## Training

On a Kaggle GPU (`gpu/train_dfine_hituav_kaggle.py`, docs/gpu_training.md), with the colour detector's recipe (batch 8, the best validation epoch kept) and its number of training steps (D-757). HIT-UAV's training set is a third the size of VisDrone's, so this means 164 epochs, not 50. A first 50-epoch run stopped while still improving, and at the evaluation's score floor it found only {{RESULT:thermal/detect_dfine_hituav_day_coco_e50_day_val.json#groups.person.ap50.max_recall|.0%}} of people on validation (person AP50 {{RESULT:thermal/detect_dfine_hituav_day_coco_e50_day_val.json#groups.person.ap50.ap|.3f}}). Three runs: from the authors' COCO weights (the thermal detector), from the VisDrone-trained colour detector (does colour aerial training help?), and from COCO weights on the official split (comparison only).

```bash
bash scripts/queues/phase13_thermal_prepare.sh   # export with parity on thermal frames, validation scores
python scripts/freeze.py --phase phase13
bash scripts/queues/phase13_thermal_test.sh      # once
```

## Thermal tracking: BIRDSAI

HIT-UAV holds single images, so tracking needs another dataset. BIRDSAI (Bondi et al., WACV 2020; LILA "Conservation Drones", Community Data License Agreement, permissive; D-754) holds 48 real thermal videos from fixed-wing drone patrols at night over African protected areas, at 60 to 120 m, with people and animals tracked frame by frame. Its cameras differ from HIT-UAV's (FLIR Vue Pro 640 or Tamarisk 640, 640 x 480, 19 mm lens, automatic gain), so it also tests whether the thermal detector carries over to another thermal camera and setting.

- 32 training videos (6 with people) tune the tracker's thresholds for people; 16 test videos (5 with people, 30 people tracked over 22,000 boxes) are scored once.
- Animals are not ignored: a person detection on an animal is a false alarm, as it would be for a search crew.
- People in these videos are small, typically 10 to 45 pixels tall.

```bash
python scripts/birdsai_prep.py
bash scripts/queues/phase13_birdsai.sh tune
bash scripts/queues/phase13_birdsai.sh test   # once
```

## Results

The one-time tests wait for the retrained thermal detector (D-757). One result is already in, from tuning on the BIRDSAI training videos, not a test.

**The colour detector barely finds people in thermal video.** The version 2 colour detector (D-FINE N, trained on VisDrone) was run on BIRDSAI's 32 training videos, and the tracker's thresholds for people were tuned by the D-733 rule. At the chosen thresholds (high {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#selected.track_high_thresh|.2f}}, new {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#selected.new_track_thresh|.2f}}), tracks covered {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#points.14.metrics.person.OVERALL.recall|.1%}} of the person boxes, and {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#points.14.metrics.person.OVERALL.precision|.1%}} of person-track boxes were people. That is {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#points.14.metrics.person.OVERALL.num_false_positives|,.0f}} false person boxes against {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#points.14.metrics.person.OVERALL.num_objects|,.0f}} true ones, with person IDF1 {{RESULT:birdsai/train/grid_dfine_n_640__motion__person.json#points.14.metrics.person.OVERALL.idf1|.3f}}. What looks like a person in colour does not carry over to thermal. The likely reason, not tested here, is that a warm body shows as a bright blob, without the shape, shading or colour the detector learned. So a search drone with only a thermal camera needs a detector trained on thermal images. Whether training on HIT-UAV is enough for BIRDSAI's different camera and setting is the one-time test still to come.

## Limits

- **One sensor.** One camera model at one resolution (640 x 512) and focal length; another sensor, lens or resolution changes how many pixels a person covers at a given height.
- **Ground temperature.** Contrast depends on the difference between a body and its surroundings. Sun-warmed ground, warm roads and buildings reduce it; cold nights increase it. Results are reported by day and night, but the dataset records no temperatures, so the effect cannot be measured directly.
- **Places and seasons.** A limited set of sites (schools, car parks, roads, playgrounds), flown on nine days in December 2020 and January 2021. Forest, open country, water and summer heat are not covered.
- **Detection only.** HIT-UAV has single images, not tracks, so thermal tracking is not scored here.
