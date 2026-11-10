# Search map (Phases 12 and 14)

Every run can be shown as a search map. It shows what was found and where, how sure each position is, and which ground the camera has already searched, next to the drone's own video.

```bash
python scripts/make_search_map.py auair --clip 35 --source cache/runs/v2_clip35
python scripts/make_search_map.py syndrone --town Town07 --height 50 --source v2_dfine
aerialrange map --run-dir cache/run --metadata <flight CSV> --origin <lat>,<lon>,<height> [--config run]
```

`make_search_map.py` writes `cache/maps/<name>/`:
- `map.html`, which opens in a browser with no internet connection;
- `map.geojson`;
- `ground.webp`, the ground image built from the flight's own frames;
- `video.mp4`.

`aerialrange map` and the ROS 2 node write the map without the video and ground image, because they do not keep the frames.

## The page

- **Video and map, side by side, on one timeline.** Every detection is boxed in the video. The colour shows the range's reliability: green reliable, amber uncertain, red no range. Click a box, a marker or a row in the list, and the same object is highlighted in all three. Selecting a find jumps the video to the moment it was first seen.
- **One marker per object**, at its most precise position: the frame whose error ellipse is smallest. The selected object shows that 95 % ellipse and its trail. Blue is people, orange is vehicles. Objects with no reliable range are hidden unless their layer is switched on.
- **Finds list.** People come first, each with when it was first seen, for how many frames, and how often its range was reliable. Each find can be marked *confirmed* or *dismissed*, so the list works as a crew's checklist. Marks are kept in that browser only.
- **Ground image.** The drone's own frames are laid flat on the ground, so the map shows the real place with no map service.
- **Searched ground.** The ground the camera has seen at a usable angle, at least 10° below the horizon (from `configs/run.yaml`). The area appears at the top of the page.
- **True positions** (SynDrone only): where the findable people and vehicles really were in each frame. Findable means in view, large enough and not hidden, the same rule the evaluation uses. So a missed person is visible on the map.
- **Simulation truth column** (SynDrone only): each find is labelled
  - **real**: its box matched a real object's;
  - **near a real one**: a real object stood inside the find's own 95 % ellipse, so a crew sent there would find it, even though the box was too poorly placed to match;
  - **false alarm**: neither.

  The page header gives the counts. On Town07 at 50 m: 7 real, 7 near and 75 false alarms among 92 people finds; 123 of 128 vehicle finds real.

People are shown as a class and a number only (build2 R16).

## The ground image

Each frame is laid onto flat ground using its camera pose. Over a flat plane, the map from image to ground is exact (a homography, fixed by four corner rays). Only the part of the image at least 10° below the horizon is used, as for the searched ground. Where frames overlap, each spot keeps the frame whose camera was nearest to it, which gives the steepest, least stretched view. `tests/test_mosaic.py` checks that each cell shows the ground it stands for, to within a cell.

It is a picture for orientation, not a measured map:
- It assumes flat ground at the pipeline's ground height, so buildings, trees and vehicles lean away from the camera.
- An error in the recorded pose moves the picture, just as it moves the ranges. On AU-AIR, whose positions and compass heading come from the drone's own sensors, neighbouring frames visibly fail to line up. On SynDrone, whose poses are exact, they join without seams.

## Map tiles

The viewer draws no outside map by default, so it sends no requests. To add one, open `map.html?tiles=<URL with {z}/{x}/{y}>&attribution=<credit text>`, using a tile service whose terms allow it. OpenStreetMap's own servers, for example, require visible attribution and forbid offline use and bulk downloads (https://operations.osmfoundation.org/policies/tiles/). The viewer asks only for the tiles in view.

SynDrone towns have no real location. They are placed at latitude 0, longitude 0 for display, and the page says so.

## Accuracy

Positions are converted to latitude and longitude exactly, with no flat-Earth approximation. The conversion agrees with an independent library to 0.1 mm. On AU-AIR, the recorded altitude is the drone's height, not a height above the ellipsoid; an error of tens of metres there moves map positions by under 1 cm per kilometre. The map is as accurate as the ranges behind it (docs/syndrone_eval.md, docs/navigation.md).

## Opening the files

Open `map.html` straight from the folder. A browser plays and seeks the local video itself. If the folder is served over a network instead, the server must support HTTP range requests, or the video cannot seek. Python's built-in `http.server` does not support them.
