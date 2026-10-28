"""Camera tilt from the ground-plane homography between two frames (D-505, D-513)."""

import numpy as np
import pytest

from aerialrange.evaluate.tilt_motion import camera_rotation, plane_homography, tilt_from_homography

K = np.array([[1259.0, 0, 960], [0, 1259.0, 540], [0, 0, 1]])


def _views(tilt_deg, move_fwd=2.0, move_right=0.5, climb=0.0, h=20.0):
    """Homography of the ground plane between two views of a camera tilted `tilt_deg` below the
    horizon (90 = straight down), translating in the world between the views."""
    R1 = camera_rotation(tilt_deg)  # world (N, E, D) -> camera
    R2 = R1
    C1 = np.array([0.0, 0.0, -h])
    C2 = C1 + np.array([move_fwd, move_right, -climb])
    return plane_homography(K, R1, C1, R2, C2)


@pytest.mark.parametrize("tilt", [90.0, 75.0, 60.0, 45.0, 30.0])
def test_tilt_recovered_from_a_synthetic_plane_homography(tilt):
    H = _views(tilt)
    est = tilt_from_homography(H, K)
    assert est is not None
    assert est == pytest.approx(tilt, abs=0.5)


def test_camera_rotation_straight_down_looks_along_down():
    R = camera_rotation(90.0)
    optical_axis_world = R.T @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(optical_axis_world, [0.0, 0.0, 1.0], atol=1e-12)


def test_camera_rotation_horizontal_looks_north():
    R = camera_rotation(0.0)
    np.testing.assert_allclose(R.T @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-12)


def test_no_motion_gives_no_estimate():
    H = _views(60.0, move_fwd=0.0, move_right=0.0)
    assert tilt_from_homography(H, K) is None
