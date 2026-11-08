"""Figure for the version 2 README: people found, by size, on the one-time VisDrone test runs.

Usage: python scripts/v2_figures.py

Reads saved results only (nothing is re-run on test data):
- tracked recall by box height, for version 1 and version 2: the share of ground-truth person
  boxes, over every frame, matched by an output track (scripts/track_breakdowns.py);
- version 2's detection recall by box height at its score floor, on every 10th frame
  (scripts/detect_eval.py), to show how many small people the detector finds before tracking.
Writes results/visdrone/test-dev/figures/people_by_size_v1_v2.png and a CSV beside it.
"""

from __future__ import annotations

import json
import sys

from aerialrange.utils.config import REPO_ROOT
from aerialrange.viz.plots import line_chart

TEST = REPO_ROOT / "results" / "visdrone" / "test-dev"
BINS = ["0-16", "16-32", "32-64", "64-128", "128-inf"]
MID = [8, 24, 48, 96, 192]  # bin centres in pixels (the last bin is open, plotted at 192)


def main() -> int:
    v1 = json.loads((TEST / "breakdowns_B2_yolo26n_960_tuned_botsort_cmc.json").read_text())
    v2 = json.loads((TEST / "breakdowns_dfine_n_640__motion_dfine_n_640.json").read_text())
    det = json.loads((TEST / "detect_dfine_n_640.json").read_text())["results"]["person"]
    hb = det["by_height"]
    series = {
        "Version 2, detected (single frames)": {
            "x": MID,
            "y": [hb[b]["max_recall"] for b in BINS],
        },
        "Version 2, tracked": {
            "x": MID,
            "y": [v2["groups"]["person"]["by_box_height"][b]["recall"] for b in BINS],
        },
        "Version 1, tracked": {
            "x": MID,
            "y": [v1["groups"]["person"]["by_box_height"][b]["recall"] for b in BINS],
        },
    }
    line_chart(
        TEST / "figures" / "people_by_size_v1_v2.png",
        series,
        "People found, by size: VisDrone test-dev",
        "Person box height (pixels)",
        "Share of people found",
        logx=True,
        xticks=[8, 16, 32, 64, 128, 256],
        note="Tracked: share of ground-truth person boxes matched by a track, every frame. "
        "Detected: version 2's detector at its score floor, every 10th frame. One-time test runs.",
    )
    print("wrote", TEST / "figures" / "people_by_size_v1_v2.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
