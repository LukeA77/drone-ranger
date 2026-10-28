"""End-to-end pipeline pieces: ranging, filter bank, run on synthetic data."""

import numpy as np
import pytest

from aerialrange.evaluate.auair_world import project_ground
from aerialrange.pipeline.filterbank import TrackFilterBank
from aerialrange.pipeline.ranging import CameraPose, RangingModel, range_boxes

MODEL = RangingModel(
    focal_px=1259.0,
    cx=960.0,
    cy=540.0,
    edges_deg=np.array([0, 10, 20, 40, 60, 91.0]),
    rel_sigma=np.array([0.217, 0.099, 0.125, 0.074, 0.038]),
    reliable_max_rel=0.10,
)


def _box_at_ground(dn, de, pose, w=40.0, h=20.0):
    u, v = project_ground(dn, de, pose.h_m, MODEL.focal_px, pose.yaw_rad, pose.tilt_deg)
    return np.array([[u - w / 2, v - h, w, h]])  # bottom-centre on the ground point


def test_range_boxes_recovers_a_known_ground_point_and_angles():
    pose = CameraPose(n_m=100.0, e_m=50.0, h_m=20.0, yaw_rad=np.radians(30), tilt_deg=45.0)
    r = range_boxes(_box_at_ground(30.0, 10.0, pose), pose, MODEL)
    assert r["valid"][0] and r["reason"][0] == "ok"
    np.testing.assert_allclose([r["ground_n_m"][0], r["ground_e_m"][0]], [130.0, 60.0], atol=1e-6)
    assert r["range_m"][0] == pytest.approx(np.sqrt(30**2 + 10**2 + 20**2))
    assert r["azimuth_deg"][0] == pytest.approx(np.degrees(np.arctan2(10, 30)) % 360)
    assert r["relative_azimuth_deg"][0] == pytest.approx(np.degrees(np.arctan2(10, 30)) - 30)
    assert r["depression_deg"][0] == pytest.approx(np.degrees(np.arctan2(20, np.hypot(30, 10))))
    band = np.searchsorted(MODEL.edges_deg[1:], r["depression_deg"][0], side="right")
    assert r["range_sigma_m"][0] == pytest.approx(MODEL.rel_sigma[band] * r["range_m"][0])
    assert r["flag"][0] == ("reliable" if MODEL.rel_sigma[band] <= 0.10 else "uncertain")


def test_range_boxes_invalid_without_tilt_and_above_the_horizon():
    no_tilt = CameraPose(0.0, 0.0, 20.0, 0.0, float("nan"))
    r = range_boxes(np.array([[900.0, 500, 40, 20]]), no_tilt, MODEL)
    assert not r["valid"][0] and r["reason"][0] == "no_tilt" and r["flag"][0] == "invalid"
    shallow = CameraPose(0.0, 0.0, 20.0, 0.0, 5.0)
    r = range_boxes(np.array([[900.0, 10, 40, 20]]), shallow, MODEL)  # bottom above the horizon
    assert not r["valid"][0] and r["reason"][0] == "above_horizon" and r["flag"][0] == "invalid"


def test_filter_bank_stationary_and_constant_velocity_targets():
    bank = TrackFilterBank(q=1.0, max_unseen=30)
    dt = 0.2
    rng = np.random.default_rng(0)
    for k in range(60):
        ids = np.array([1, 2])
        z = np.array([[5.0, 5.0], [0.0 + 3.0 * k * dt, 10.0]]) + rng.normal(0, 0.05, (2, 2))
        sig = np.array([0.05, 0.05])
        out = bank.step(dt, ids, z, sig)
    sp = dict(zip(out["id"].tolist(), out["speed_ms"].tolist(), strict=True))
    sig = dict(zip(out["id"].tolist(), out["speed_sigma_ms"].tolist(), strict=True))
    # a stationary object's speed is consistent with zero given its sigma
    assert sp[1] < 3 * sig[1]
    # the mover goes 3 m/s north; its estimate lies within 3 sigma
    assert abs(sp[2] - 3.0) < 3 * sig[2]


def test_filter_bank_predicts_through_misses_and_drops_lost_tracks():
    bank = TrackFilterBank(q=1.0, max_unseen=3)
    for k in range(5):
        bank.step(0.2, np.array([7]), np.array([[0.0, 2.0 * k * 0.2]]), np.array([0.05]))
    out = bank.step(0.2, np.array([], int), np.zeros((0, 2)), np.zeros(0))
    assert 7 in out["id"].tolist()  # still alive, predicted
    for _ in range(3):
        out = bank.step(0.2, np.array([], int), np.zeros((0, 2)), np.zeros(0))
    assert 7 not in out["id"].tolist()  # dropped after max_unseen frames without a measurement


def test_demo_runs_end_to_end_and_matches_truth(tmp_path):
    pytest.importorskip("ultralytics")
    import json

    import pandas as pd

    from aerialrange.pipeline.demo import demo
    from aerialrange.pipeline.run import COLUMNS

    s = demo(tmp_path, n_frames=30)
    tr = pd.read_csv(tmp_path / "run" / "tracks.csv")
    assert list(tr.columns) == COLUMNS  # every output field
    assert (tmp_path / "run" / "annotated.mp4").stat().st_size > 0
    timing = json.loads((tmp_path / "run" / "timing.json").read_text())
    assert timing["frames"] == 30 and "end_to_end" in timing["stages"]
    assert s["valid_fraction"] > 0.9
    # range to the box bottom (nearest footprint edge) reads a little short of the centre
    assert s["range_rel_error_median"] < 0.08
    assert set(tr["flag"]) <= {"reliable", "uncertain", "invalid"}
