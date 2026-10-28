# Superseded results

These three validation runs of tiled detection (baseline B3) are **invalid** and kept only for the record. They were run before a bug fix: the detections passed to BoT-SORT lacked the corner-format boxes its camera-motion compensation uses to mask moving objects, so compensation silently fell back to no correction (`gmc_off`). Their scores understate B3.

The corrected run is `../B3_yolo26n_640_botsort_cmc_ios0.5.json`. The fix is in `src/aerialrange/track/runner.py` (`DetSet.xyxy`) and is described in `docs/decisions.md`, D-627.
