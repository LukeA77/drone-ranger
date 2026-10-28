"""Mid-Air loader tests on a tiny SYNTHETIC fixture (HDF5 + PNG), no real data needed."""

import h5py
import numpy as np
import pytest
from PIL import Image

from aerialrange.data import midair as mid
from aerialrange.geometry import frames as fr


def _write_depth(path, depth_m):
    bits = np.asarray(depth_m, np.float16).view(np.uint16)
    Image.fromarray(bits).save(path)


@pytest.fixture
def climate(tmp_path):
    """Two-frame trajectory with known pose, depth, segmentation and normals."""
    root = tmp_path / "Kite_training" / "sunny"
    n_frames, gt_per_frame = 2, 4
    q = fr.R_to_quat(fr.R_wb_from_ypr(0.3, -0.1, 0.05))  # wxyz
    root.mkdir(parents=True)
    with h5py.File(root / "sensor_records.hdf5", "w") as f:
        t = f.create_group("trajectory_0000")
        cd = t.create_group("camera_data")
        for mod, ext in (("depth", "PNG"), ("segmentation", "PNG"), ("normals", "PNG")):
            cd[mod] = np.array(
                [f"{mod}/trajectory_0000/{k:06d}.{ext}".encode() for k in range(n_frames)]
            )
        gt = t.create_group("groundtruth")
        n_gt = n_frames * gt_per_frame
        pos = gt.create_dataset("position", data=np.arange(n_gt * 3, dtype=float).reshape(n_gt, 3))
        pos.attrs["sampling_frequency"] = 100.0
        att = gt.create_dataset("attitude", data=np.tile(q, (n_gt, 1)))
        att.attrs["axis_ordering"] = "wxyz"
        att.attrs["sampling_frequency"] = 100.0
    for mod in ("depth", "segmentation", "normals"):
        (root / mod / "trajectory_0000").mkdir(parents=True)
    for k in range(n_frames):
        _write_depth(
            root / f"depth/trajectory_0000/{k:06d}.PNG", np.array([[10.0, 20.5], [65504.0, 1.25]])
        )
        Image.fromarray(np.array([[0, 4], [8, 10]], np.uint8)).save(
            root / f"segmentation/trajectory_0000/{k:06d}.PNG"
        )
        flat = np.array([[[128, 255, 128]]], np.uint8)  # level flat ground
        Image.fromarray(flat).save(root / f"normals/trajectory_0000/{k:06d}.PNG")
    return root, q


def test_list_trajectories(climate):
    root, _ = climate
    assert mid.list_trajectories(root / "sensor_records.hdf5") == ["trajectory_0000"]


def test_depth_decoding_reinterprets_float16_bits(climate):
    root, _ = climate
    d = mid.read_depth(root / "depth/trajectory_0000/000000.PNG")
    np.testing.assert_array_equal(d, [[10.0, 20.5], [65504.0, 1.25]])
    assert d.dtype == np.float32


def test_sky_mask(climate):
    root, _ = climate
    d = mid.read_depth(root / "depth/trajectory_0000/000000.PNG")
    np.testing.assert_array_equal(mid.sky_mask(d), [[False, False], [True, False]])


def test_segmentation_is_class_ids(climate):
    root, _ = climate
    s = mid.read_segmentation(root / "segmentation/trajectory_0000/000000.PNG")
    np.testing.assert_array_equal(s, [[0, 4], [8, 10]])


def test_normals_decoding_level_flat_ground_points_up_in_body():
    rgb = np.array([[[128, 255, 128]]], np.uint8)
    n = mid.decode_normals(rgb)
    np.testing.assert_allclose(n[0, 0], [0, 0, -1], atol=0.01)  # body up = -z (FRD)
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0)


def test_normals_channel_mapping():
    # Red -> body y, blue -> body x, green -> -body z (Mid-Air technical specifications).
    n = mid.decode_normals(np.array([[[255, 128, 128]]], np.uint8))
    assert n[0, 0, 1] > 0.99
    n = mid.decode_normals(np.array([[[128, 128, 255]]], np.uint8))
    assert n[0, 0, 0] > 0.99
    n = mid.decode_normals(np.array([[[128, 0, 128]]], np.uint8))
    assert n[0, 0, 2] > 0.99


def test_camera_intrinsics_and_mount():
    cam = mid.midair_camera(1024, 1024)
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (512.0, 512.0, 512.0, 512.0)
    np.testing.assert_array_equal(cam.R_bc, fr.R_BC_FORWARD)


def test_trajectory_frame_to_groundtruth_row(climate):
    root, q = climate
    traj = mid.MidAirTrajectory(root, "trajectory_0000")
    assert traj.n_frames == 2
    assert traj.gt_row(0) == 0 and traj.gt_row(1) == 4
    pose = traj.pose(1)
    np.testing.assert_allclose(pose.p_w, np.arange(12, 15, dtype=float))
    np.testing.assert_allclose(pose.R_wb, fr.quat_to_R(q), atol=1e-12)
    assert traj.frame_path("depth", 1).name == "000001.PNG"
    traj.close()


def test_trajectory_rejects_unknown_quaternion_order(climate):
    root, _ = climate
    with h5py.File(root / "sensor_records.hdf5", "a") as f:
        f["trajectory_0000/groundtruth/attitude"].attrs["axis_ordering"] = "xyzw"
    with pytest.raises(ValueError, match="wxyz"):
        mid.MidAirTrajectory(root, "trajectory_0000")


def test_radial_depth_back_projection_has_norm_equal_to_depth():
    cam = mid.midair_camera(8, 8)
    uv = np.array([[0.0, 0.0], [4.0, 4.0], [7.0, 2.0]])
    depth = np.array([10.0, 20.0, 5.0])
    X = mid.backproject(cam, uv, depth, kind="radial")
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), depth)
    np.testing.assert_allclose(X[1], [0, 0, 20.0])  # principal point


def test_planar_depth_back_projection_has_z_equal_to_depth():
    cam = mid.midair_camera(8, 8)
    uv = np.array([[0.0, 0.0], [7.0, 2.0]])
    X = mid.backproject(cam, uv, np.array([10.0, 5.0]), kind="planar")
    np.testing.assert_allclose(X[:, 2], [10.0, 5.0])
    with pytest.raises(ValueError):
        mid.backproject(cam, uv, np.array([1.0, 1.0]), kind="spherical")


def test_world_points_from_depth_round_trip(climate):
    """Ground-truth pixel in world projects back to the same pixel."""
    from aerialrange.geometry.camera import Pose, project_world_point

    cam = mid.midair_camera(64, 64)
    pose = Pose.from_ypr([1.0, 2.0, -30.0], 0.4, -0.2, 0.1)
    uv = np.array([[10.0, 50.0], [40.0, 33.0]])
    X_w = mid.world_points(cam, pose, uv, np.array([25.0, 40.0]), kind="radial")
    for p, x in zip(uv, X_w, strict=True):
        back, ok = project_world_point(cam, pose, x)
        assert ok
        np.testing.assert_allclose(back, p, atol=1e-9)
