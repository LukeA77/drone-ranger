"""`aerialrange demo`: the whole pipeline on a synthetic flight, checked against its truth.

No datasets or model weights are needed: boxes come from the synthetic scene (the detector
stage reads them instead of running a model), so the run exercises tracking, ranging, the
filter bank, the overlay and the outputs. Ground-truth positions and speeds are known by
construction, so the demo reports how close the estimates are. Synthetic data, labelled as
such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aerialrange.pipeline.run import run
from aerialrange.pipeline.synthetic import make_flight


def demo(out: Path, n_frames: int = 60) -> dict[str, Any]:
    out = Path(out)
    scene = make_flight(out / "flight", n_frames=n_frames)
    overrides = {
        "camera": {"focal_px": scene["focal_px"], "cx": scene["cx"], "cy": scene["cy"]},
        "detector": {"kind": "boxes"},
        "overlay": {"font_scale": 0.45},
    }
    timing = run(
        out / "flight" / "frames",
        out / "flight" / "metadata.csv",
        out / "run",
        boxes_csv=out / "flight" / "boxes.csv",
        overrides=overrides,
        credit="SYNTHETIC demo scene (aerialrange)",
    )
    tr = pd.read_csv(out / "run" / "tracks.csv")
    truth = pd.read_csv(out / "flight" / "truth.csv")
    meta = pd.read_csv(out / "flight" / "metadata.csv").set_index("frame")
    errs, speeds = [], {}
    for _, r in tr[tr["valid"]].iterrows():
        cam = meta.loc[r["frame"]]
        t = truth[truth["frame"] == r["frame"]]
        if not len(t):
            continue
        true_rng = np.hypot(np.hypot(t["n_m"] - cam["n_m"], t["e_m"] - cam["e_m"]), cam["h_m"])
        j = int(np.argmin(np.abs(true_rng.to_numpy() - r["range_m"])))
        errs.append((r["range_m"] - true_rng.iloc[j]) / true_rng.iloc[j])
        if np.isfinite(r["speed_ms"]):
            speeds.setdefault(int(t["car"].iloc[j]), []).append(r["speed_ms"])
    summary = {
        "synthetic": True,
        "frames": timing["frames"],
        "rows": int(len(tr)),
        "valid_fraction": float(tr["valid"].mean()) if len(tr) else 0.0,
        "range_rel_error_median": float(np.median(np.abs(errs))) if errs else None,
        "range_rel_error_signed_median": float(np.median(errs)) if errs else None,
        "speed_last_10_median_ms": {k: float(np.median(v[-10:])) for k, v in speeds.items()},
        "true_speeds_ms": dict(enumerate(scene["speeds_ms"], start=1)),
        "note": "Range is to the box bottom-centre, the nearest edge of each footprint, so it "
        "reads slightly short of the footprint centre by construction.",
    }
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
