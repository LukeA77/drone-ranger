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


def test_footprint_correction_moves_ranges_along_the_camera_heading():
    meta, tracks = scene()
    sizes = {
        "vehicle": {
            "length_mean_m": 4.6,
            "width_mean_m": 1.9,
            "length_var_m2": 0.0,
            "width_var_m2": 0.0,
            "length_width_cov_m2": 0.0,
        }
    }
    plain = replay(tracks, meta, RangeSpeedStage(MODEL, REL, SD))
    moved = replay(tracks, meta, RangeSpeedStage(MODEL, REL, SD, {"enabled": True, "sizes": sizes}))
    off = moved["range_m"] - plain["range_m"]
    assert (off > 0).all()  # the camera looks north, ahead of it: centres are further away
    # first frame: no filter heading yet, so the offset along the heading is (L + W) / pi
    d_first = (4.6 + 1.9) / np.pi
    dep = np.radians(plain["depression_deg"].iloc[0])
    assert off.iloc[0] == pytest.approx(d_first * np.cos(dep), rel=0.05)
    assert (moved["range_sigma_m"] >= plain["range_sigma_m"]).all()
    off_disabled = replay(
        tracks, meta, RangeSpeedStage(MODEL, REL, SD, {"enabled": False, "sizes": sizes})
    )
    np.testing.assert_allclose(off_disabled["range_m"], plain["range_m"])


def test_calibrated_half_widths_scale_the_sigmas():
    meta, tracks = scene()
    cal = {
        "dep_edges": [0, 20, 40, 60, 91],
        "range_edges": [0, 50, 100, 200, np.inf],
        "range": {"pooled": {"0.68": 1.0, "0.90": 2.0, "0.95": 3.0}},
        "speed": {"pooled": {"0.68": 1.5, "0.90": 2.5, "0.95": 3.5}},
    }
    out = replay(tracks, meta, RangeSpeedStage(MODEL, REL, SD, None, cal))
    np.testing.assert_allclose(out["range_half_90_m"], 2 * out["range_sigma_m"])
    sp = out.dropna(subset=["speed_sigma_ms"])
    np.testing.assert_allclose(sp["speed_half_95_ms"], 3.5 * sp["speed_sigma_ms"])
    plain = replay(tracks, meta, RangeSpeedStage(MODEL, REL, SD))
    assert plain["range_half_90_m"].isna().all()


def test_display_rules_per_group_and_one_sided_nis():
    from aerialrange.pipeline.range_speed import display_rules
    from aerialrange.pipeline.speed_display import SpeedDisplayRule

    sd = {
        "sigma_max_ms": None,
        "min_age": None,
        "nis_window": 10,
        "nis_conf": 0.95,
        "nis_test": "upper",
        "by_group": {"person": {"sigma_max_ms": 0.2, "min_age": 15}, "vehicle": None},
    }
    rules = display_rules(sd)
    assert rules["vehicle"] is None and rules["person"].nis_test == "upper"
    r = rules["person"]
    assert r.ok(0.1, 20, 0.05, 10)  # a calm, conservative filter passes the one-sided test
    assert not SpeedDisplayRule(0.2, 15, 10, 0.95).ok(0.1, 20, 0.05, 10)  # two-sided rejects
    assert not r.ok(0.1, 20, 5.0, 10)  # an overconfident filter fails
