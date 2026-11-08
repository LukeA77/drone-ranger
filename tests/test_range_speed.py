"""The ranging and speed stage after the tracker (pipeline/range_speed.py)."""

import numpy as np
import pandas as pd
import pytest

from aerialrange.pipeline.range_speed import RangeSpeedStage, replay
from aerialrange.pipeline.ranging import CameraPose, RangingModel

MODEL = RangingModel(
    1000.0, 960.0, 540.0, np.array([0, 10, 20, 40, 60, 91.0]), np.full(5, 0.05), 0.10
)
SD = {"sigma_max_ms": None, "min_age": None, "nis_window": 10, "nis_conf": 0.95}
REL = {"q": 1.0, "meas_rel_sigma": [0.01] * 5, "max_unseen_frames": 30}
PROP = {
    "q": 1.0,
    "max_unseen_frames": 30,
    "meas_model": "propagated",
    "meas_sigma": {
        "kappa_u": {"vehicle": 0.04},
        "kappa_v": {"vehicle": 0.05},
        "tilt_deg": 0.5,
        "h_m": 0.0,
        "yaw_deg": 0.0,
    },
}


def scene(n=12):
    """A car crossing the image under a camera 40 m up, tilted 30 degrees."""
    meta = pd.DataFrame(
        {
            "frame": np.arange(1, n + 1),
            "t_s": np.arange(n) * 0.2,
            "n_m": 0.0,
            "e_m": 0.0,
            "h_m": 40.0,
            "yaw_rad": 0.0,
            "tilt_deg": 30.0,
        }
    ).set_index("frame")
    tracks = pd.DataFrame(
        {
            "frame": np.arange(1, n + 1),
            "track_id": 7,
            "class": "vehicle",
            "left": 900.0 + 10 * np.arange(n),
            "top": 600.0,
            "width": 60.0,
            "height": 40.0,
        }
    )
    return meta, tracks


def test_replay_gives_one_row_per_box_and_a_speed_after_two_frames():
    meta, tracks = scene()
    out = replay(tracks, meta, RangeSpeedStage(MODEL, REL, SD))
    assert len(out) == len(tracks) and out["valid"].all()
    assert np.isnan(out["speed_ms"].iloc[0]) and np.isfinite(out["speed_ms"].iloc[-1])


def test_propagated_errors_are_anisotropic_and_grow_with_tilt_sigma():
    meta, tracks = scene(1)
    st = RangeSpeedStage(MODEL, PROP, SD)
    pose = CameraPose(0.0, 0.0, 40.0, 0.0, 30.0)
    boxes = tracks[["left", "top", "width", "height"]].to_numpy()
    from aerialrange.pipeline.ranging import range_boxes

    rg = range_boxes(boxes, pose, MODEL)
    c1 = st.measurement_errors(boxes, ["vehicle"], rg, pose, np.nan)[0]
    c2 = st.measurement_errors(boxes, ["vehicle"], rg, pose, 2.0)[0]  # per-frame tilt sigma
    ev = np.linalg.eigvalsh(c1)
    assert ev[1] > 4 * ev[0]  # longer along the line of sight
    assert np.trace(c2) > np.trace(c1)
    with pytest.raises(ValueError):
        RangeSpeedStage(MODEL, {**PROP, "meas_model": "guess"}, SD)


def test_kappa_by_box_height_uses_the_bin_and_falls_back_on_empty_bins():
    from aerialrange.pipeline.range_speed import kappa_at

    spec = {
        "edges_px": [0, 16, 32, 64, 128, np.inf],
        "values": [np.nan, 0.11, 0.08, 0.05, 0.03],
        "overall": 0.06,
    }
    assert kappa_at(0.04, 50.0) == 0.04
    assert kappa_at(spec, 20.0) == 0.11 and kappa_at(spec, 500.0) == 0.03
    assert kappa_at(spec, 8.0) == 0.06  # empty bin
