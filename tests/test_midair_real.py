"""Regression tests on REAL Mid-Air data.

Skipped when the data is absent (CI has no datasets). Thresholds sit well outside the values
measured on the development trajectories (results/midair/unknowns.json), so they
catch a wrong convention, not small changes.
"""

import numpy as np
import pytest

from aerialrange.evaluate import midair_unknowns as unk
from aerialrange.utils.config import REPO_ROOT, load_config

_data = load_config("data")
ROOT = REPO_ROOT / _data.get("data_root") / _data.get("midair_subdir") / "Kite_training" / "sunny"
NAME = "trajectory_0002"  # development split (results/splits.json)
pytestmark = pytest.mark.skipif(
    not (ROOT / "sensor_records.hdf5").exists()
    or not all(
        (ROOT / m / NAME / ".frames.done").exists() for m in ("depth", "segmentation", "normals")
    ),
    reason="Mid-Air Kite_training/sunny trajectory_0002 not downloaded",
)


def test_quaternion_rotates_body_to_world():
    r = unk.quaternion_check(ROOT / "sensor_records.hdf5", NAME)
    assert r["R_wb_body_to_world"] < 0.1 * r["R_bw_world_to_body"]


def test_depth_is_radial_and_water_is_level():
    rows = unk.water_plane_check(ROOT, NAME, stride=200, min_pixels=2000, max_pixels=3000, seed=0)
    assert len(rows) >= 3
    radial = np.median([r["radial_rms_m"] for r in rows])
    planar = np.median([r["planar_rms_m"] for r in rows])
    assert radial < 0.05 * planar
    assert np.median([r["radial_tilt_deg"] for r in rows]) < 0.5


def test_decoded_normals_agree_with_attitude_on_average():
    rows = unk.normals_check(ROOT, NAME, stride=200, min_pixels=500)
    assert len(rows) >= 3
    assert np.median([r["mean_normal_tilt_deg"] for r in rows]) < 1.0


def test_sky_sentinel_matches_class_zero():
    r = unk.sky_check(ROOT, NAME, stride=200)
    assert r["sky_not_class0"] + r["class0_not_sky"] < 1e-3 * r["sky_and_class0"]


def test_plane_fit_on_synthetic_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-10, 10, size=(200, 2))
    X = np.column_stack([xy, 0.1 * xy[:, 0] + 3.0])
    n, rms = unk.fit_plane(X)
    assert rms < 1e-10
    np.testing.assert_allclose(abs(n @ [-0.1, 0, 1]) / np.linalg.norm([-0.1, 0, 1]), 1.0)
