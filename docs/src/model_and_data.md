# Model and data

The detector and tracker, and the three datasets, with their licences.

## Model: detector and tracker

### What it is

- **Detector:** YOLO26n (Ultralytics 8.4.182), fine-tuned from Common Objects in Context (COCO) weights on the VisDrone2019 detection training set at an input size of {{RESULT:headline.json#model.imgsz|d}} pixels, for {{RESULT:headline.json#model.epochs|d}} epochs on one {{RESULT:headline.json#model.gpu|s}} graphics processing unit (GPU) ({{RESULT:headline.json#model.train_hours|.1f}} hours, library defaults otherwise). The best epoch was the last ({{RESULT:headline.json#model.best_epoch|d}}), so the model had not finished improving.
- **Tracker:** BoT-SORT from the same library, motion and overlap only (no appearance re-identification, by design), with camera-motion compensation and two confidence thresholds tuned on the validation set (D-321).
- **Exported form:** Open Neural Network Exchange (ONNX); its detections match the original exactly on the validation frames checked (D-604).

### Intended use

Research evaluation of detection, tracking and ground-plane distance estimation from low-altitude drone video of road scenes. Not intended for identifying individuals, for safety-critical decisions, or for use outside the tested conditions (see limitations).

### Classes

Trained on VisDrone's ten classes; evaluated as two groups: **person** (pedestrian, people) and **vehicle** (car, van, truck, bus).

### Performance

| Data | Measure | Person | Vehicle |
|---|---|---|---|
| VisDrone detection validation set (all ten classes) | mAP50 / mAP50-95 | {{RESULT:headline.json#model.det_val_map50|.3f}} / {{RESULT:headline.json#model.det_val_map50_95|.3f}} | (all classes) |
| VisDrone MOT validation frames | AP50 | {{RESULT:headline.json#model.mot_val_person_ap50|.3f}} | {{RESULT:headline.json#model.mot_val_vehicle_ap50|.3f}} |
| VisDrone MOT test-dev (run once) | AP50 | {{RESULT:headline.json#visdrone_test.person_ap50|.3f}} | {{RESULT:headline.json#visdrone_test.vehicle_ap50|.3f}} |
| VisDrone MOT test-dev (run once) | Tracking IDF1 / MOTA | {{RESULT:headline.json#visdrone_test.person_idf1|.3f}} / {{RESULT:headline.json#visdrone_test.person_mota|.3f}} | {{RESULT:headline.json#visdrone_test.vehicle_idf1|.3f}} / {{RESULT:headline.json#visdrone_test.vehicle_mota|.3f}} |
| AU-AIR held-out flights (cross-dataset) | AP50 at IoU 0.5 | {{RESULT:headline.json#auair.heldout_person_ap50|.3f}} | {{RESULT:headline.json#auair.heldout_vehicle_ap50|.3f}} |

AP50: average precision at an intersection over union (IoU) of 0.5. mAP50-95: mean over IoU 0.5 to 0.95. IDF1: identity F1 score. MOTA: multiple object tracking accuracy.

### Known weaknesses

- Small objects: people 16 to 32 pixels tall are tracked {{RESULT:headline.json#visdrone_test.person_recall_16_32|.0%}} of the time on test-dev.
- Night scenes, partial occlusion and dense queues of similar vehicles cause most misses and identity switches (failure gallery, `results/failures/`).
- On AU-AIR (different camera, many straight-down views) detection transfers poorly at the protocol threshold; AU-AIR's looser boxes explain part of the gap (D-512).

### Training data and licence

VisDrone2019 detection train (with {{RESULT:headline.json#data.visdrone_excluded_images|d}} images removed that share source videos with the validation and test-dev tracking clips, D-308). VisDrone states no licence; the weights are not distributed. The code is AGPL-3.0-only, matching Ultralytics.

## Data

*No data is redistributed in this repository. File counts and schema checks come from `results/data_check.json`. Per-file checksums are not recorded: `scripts/data_manifest.py` computes them, but reading the full datasets takes hours and it was not run.*

### VisDrone2019 (tracking and detection)

- **Source:** https://github.com/VisDrone/VisDrone-Dataset. Collected by the AISKYEYE team, Lab of Machine Learning and Data Mining, Tianjin University.
- **Citation:** Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q. and Ling, H. (2021) 'Detection and tracking meet drones challenge', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), pp. 7380-7399. To verify against the journal before publication.
- **Licence and terms (checked 2026-10-19):** no licence file and no stated terms. Used for research evaluation only; not redistributed; only code, metrics and figures are published. Thumbnails in the failure gallery and frames in the demo video are kept minimal and credited.
- **Used:** tracking (MOT) train, validation and test-dev; detection (DET) train and validation.
- **Splits:** DET train for training; DET validation for model selection; MOT validation for tracker tuning; MOT test-dev once, at the end.
- **Known issue (D-308):** VisDrone's statement that its image set does not overlap its videos is false; {{RESULT:headline.json#data.visdrone_excluded_images|d}} overlapping detection images were removed from training and selection.

### Mid-Air (distance ground truth, synthetic)

- **Source:** https://midair.ulg.ac.be; scripts at https://github.com/montefiore-ai/midair-dataset.
- **Citation:** Fonder, M. and Van Droogenbroeck, M. (2019) 'Mid-Air: a multi-modal dataset for extremely low altitude drone flights', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*. To verify before publication.
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). Not redistributed.
- **Used:** one climate (Kite_training, sunny), {{RESULT:headline.json#data.midair_total|d}} flights: depth, segmentation, surface normals and colour images of the left camera, plus flight records.
- **Splits:** by flight, seed 0: {{RESULT:headline.json#data.midair_dev|d}} development, {{RESULT:headline.json#data.midair_cal|d}} calibration, {{RESULT:headline.json#data.midair_test|d}} test (`results/splits.json`).
- **Limitation:** synthetic, with no people or vehicles; it validates the geometry, not the detector.

### AU-AIR (real drone footage)

- **Source:** https://bozcani.github.io/auairdataset (authors' page); also indexed at https://fmi-data-index.github.io/au_air.html.
- **Citation:** Bozcan, I. and Kayacan, E. (2020) 'AU-AIR: a multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance', *IEEE International Conference on Robotics and Automation (ICRA)*. To verify before publication.
- **Licence:** the third-party index states CC BY 4.0; The authors' annotation file lists CC BY-NC-SA 2.0 and CC BY-NC 2.0. The stricter reading is used: non-commercial research, attributed, and no AU-AIR images in this repository (D-323).
- **Used:** all {{RESULT:headline.json#data.auair_frames|,d}} annotated frames (5 frames per second, {{RESULT:headline.json#data.auair_recordings|d}} recordings, {{RESULT:headline.json#data.auair_clips|d}} clips), with per-frame GPS position, altitude and drone attitude.
- **Splits:** calibration recordings 20190829091111 and 20190905142119 for the camera model; the other six held out (D-506).
- **Findings about the data:** altitude is in millimetres and drifts by metres within a flight; the camera tilt is not recorded; annotation boxes are looser than tight boxes (D-512) and not consistent from frame to frame (D-516); the paper's stated camera angles (45-90°) do not cover all clips (D-504).
