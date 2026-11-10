# Map output (Phase 12)

Every run can be shown on a map: where each tracked person or vehicle was, how sure the position is, and which ground the camera has already searched.

```bash
aerialrange map --run-dir cache/run --metadata <flight CSV> --origin <lat>,<lon>,<height> [--config run]
python scripts/auair_map.py --clip 35 --source cache/runs/v2_clip35   # an AU-AIR clip, origin from its first GPS fix
```

Both write `map.geojson` and `map.html` into the run folder. Open `map.html` in a browser; it needs no internet connection.

## What the map shows

| Layer | Meaning |
|---|---|
| Searched ground | The ground the camera has seen so far, out to where a range can still be reliable (10° below the horizon, from `configs/run.yaml`). The current frame's view is outlined. |
| Tracks | Each object's reported positions over time; blue for people, orange for vehicles. |
| Error ellipses | Where the object is, with 95 % confidence, for the tracks seen in the current frame. The ellipse matches the run's calibrated range interval. |
| Drone path | The drone's own track. |

Click an object for its class, track number, range with its interval, reliability, and speed and heading when the display rule shows them. People are shown as a class and a number only.

## Map tiles

The viewer draws no background map by default, so it sends no requests. To add one, open it as `map.html?tiles=<URL with {z}/{x}/{y}>&attribution=<credit text>` with a tile service whose terms allow it. OpenStreetMap's own servers, for example, require the attribution to be visible and forbid offline use and bulk downloads (https://operations.osmfoundation.org/policies/tiles/). The viewer asks only for the tiles in view.

## Accuracy

Positions are converted to latitude and longitude exactly (no flat-Earth approximation). The conversion agrees with an independent library to 0.1 mm. On AU-AIR, the recorded altitude is the drone's height, not a height above the ellipsoid; errors of tens of metres there move map positions by under 1 cm per kilometre. The map is as accurate as the ranges behind it (docs/syndrone_eval.md, docs/navigation.md).
