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

SPEED_TOL_MS = (
    0.5  # same tolerance as the display fit (D-706 / scripts/syndrone_speed_display_fit.py)
)


def demo(
    out: Path, n_frames: int = 60, speed_display: dict[str, Any] | None = None
) -> dict[str, Any]:
    """`speed_display` overrides the configured display rule (tests; the configured rule is
    used otherwise, and shows nothing while `sigma_max_ms` is unset (D-710))."""
    out = Path(out)
    scene = make_flight(out / "flight", n_frames=n_frames)
    overrides = {
        "camera": {"focal_px": scene["focal_px"], "cx": scene["cx"], "cy": scene["cy"]},
        "detector": {"kind": "boxes"},
        "overlay": {"font_scale": 0.45},
    }
    if speed_display is not None:
        overrides["speed_display"] = speed_display
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
    errs, speeds, shown_err, shown_sig = [], {}, [], []
    for _, r in tr[tr["valid"]].iterrows():
        cam = meta.loc[r["frame"]]
        t = truth[truth["frame"] == r["frame"]]
        if not len(t):
            continue
        true_rng = np.hypot(np.hypot(t["n_m"] - cam["n_m"], t["e_m"] - cam["e_m"]), cam["h_m"])
        # Match the row to the car nearest its estimated ground point (range, depression and
        # azimuth); matching on range alone swaps cars at similar distances.
        horiz = r["range_m"] * np.cos(np.radians(r["depression_deg"]))
        gn = cam["n_m"] + horiz * np.cos(np.radians(r["azimuth_deg"]))
        ge = cam["e_m"] + horiz * np.sin(np.radians(r["azimuth_deg"]))
        j = int(np.argmin(np.hypot(t["n_m"].to_numpy() - gn, t["e_m"].to_numpy() - ge)))
        errs.append((r["range_m"] - true_rng.iloc[j]) / true_rng.iloc[j])
        car = int(t["car"].iloc[j])
        if np.isfinite(r["speed_ms"]):
            speeds.setdefault(car, []).append(r["speed_ms"])
        if bool(r.get("speed_shown", False)):
            shown_err.append(r["speed_ms"] - scene["speeds_ms"][car - 1])
            shown_sig.append(r["speed_sigma_ms"])
    summary = {
        "synthetic": True,
        "frames": timing["frames"],
        "rows": int(len(tr)),
        "valid_fraction": float(tr["valid"].mean()) if len(tr) else 0.0,
        "range_rel_error_median": float(np.median(np.abs(errs))) if errs else None,
        "range_rel_error_signed_median": float(np.median(errs)) if errs else None,
        "speed_last_10_median_ms": {k: float(np.median(v[-10:])) for k, v in speeds.items()},
        "true_speeds_ms": dict(enumerate(scene["speeds_ms"], start=1)),
        "displayed_speed": {
            "rule": timing.get("speed_display"),
            "n_shown": len(shown_err),
            "share_of_valid_rows": len(shown_err) / max(int(tr["valid"].sum()), 1),
            "median_abs_error_ms": float(np.median(np.abs(shown_err))) if shown_err else None,
            "max_abs_error_ms": float(np.max(np.abs(shown_err))) if shown_err else None,
            "tolerance_ms": SPEED_TOL_MS,
            "share_within_tolerance": float(np.mean(np.abs(shown_err) <= SPEED_TOL_MS))
            if shown_err
            else None,
            "share_within_own_95_interval": float(
                np.mean(np.abs(shown_err) <= 1.96 * np.asarray(shown_sig))
            )
            if shown_err
            else None,
        },
        "note": "Range is to the box bottom-centre, the nearest edge of each footprint, so it "
        "reads slightly short of the footprint centre by construction.",
    }
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
