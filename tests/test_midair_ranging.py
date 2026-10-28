"""Mid-Air ranging evaluation on a SYNTHETIC flat-ground frame with exact radial depth."""

import h5py
import numpy as np
import pytest
from PIL import Image

from aerialrange.data.midair import MidAirTrajectory, midair_camera
from aerialrange.evaluate.midair_ranging import (
    Sampling,
    candidate_mask,
    discontinuity_mask,
    evaluate_trajectory,
)
from aerialrange.geometry import frames as fr
from aerialrange.geometry.camera import Pose, pixel_to_world_ray

W = 64
S = Sampling(
    pix_per_frame=200, frame_stride=1, edge_margin_px=1, depth_edge_rel=0.1, normal_patch_px=3
)  # synthetic test settings


def _render_flat(pose, z_ground):
    """Radial depth of the plane Down = z_ground (sky above the horizon)."""
    cam = midair_camera(W, W)
    vv, uu = np.mgrid[0:W, 0:W]
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(float)
    C, d = pixel_to_world_ray(cam, pose, uv)
    lam = (z_ground - C[2]) / d[:, 2]
    depth = np.where(lam > 0, lam * np.linalg.norm(d, axis=1), 65504.0).reshape(W, W)
    return np.minimum(depth, 65504.0).astype(np.float16)


@pytest.fixture
def flat_climate(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    n = 3
    poses = [Pose.from_ypr([2.0 * k, 0.0, -30.0 - k], 0.2 * k, -0.6, 0.05) for k in range(n)]
    with h5py.File(root / "sensor_records.hdf5", "w") as f:
        g = f.create_group("trajectory_0000")
        for mod in ("depth", "segmentation", "normals"):
            g[f"camera_data/{mod}"] = np.array(
                [f"{mod}/trajectory_0000/{k:06d}.PNG".encode() for k in range(n)]
            )
            (root / mod / "trajectory_0000").mkdir(parents=True)
        rows = np.repeat(np.arange(n), 4)
        pos = g.create_dataset("groundtruth/position", data=np.array([poses[r].p_w for r in rows]))
        pos.attrs["sampling_frequency"] = 100.0
        att = g.create_dataset(
            "groundtruth/attitude", data=np.array([fr.R_to_quat(poses[r].R_wb) for r in rows])
        )
        att.attrs["axis_ordering"] = "wxyz"
    for k, pose in enumerate(poses):
        depth = _render_flat(pose, z_ground=0.0)
        Image.fromarray(depth.view(np.uint16)).save(root / f"depth/trajectory_0000/{k:06d}.PNG")
        seg = np.where(depth >= 65504, 0, 4).astype(np.uint8)
        Image.fromarray(seg).save(root / f"segmentation/trajectory_0000/{k:06d}.PNG")
        up_b = pose.R_wb.T @ [0, 0, -1.0]  # level ground normal in body frame
        rgb = np.round((np.array([up_b[1], -up_b[2], up_b[0]]) + 1) / 2 * 255).astype(np.uint8)
        Image.fromarray(np.tile(rgb, (W // 2, W // 2, 1))).save(
            root / f"normals/trajectory_0000/{k:06d}.PNG"
        )
    return root


def test_discontinuity_mask_marks_jumps_and_sky():
    d = np.full((5, 5), 10.0)
    d[:, 3:] = 20.0
    d[0, 0] = 65504.0
    seg = np.full((5, 5), 4, np.uint8)
    m = discontinuity_mask(d, seg, rel=0.1)
    assert m[:, 2].all() and m[:, 3].all() and m[0, 0]
    assert not m[2, 0]


def test_candidate_mask_respects_margin():
    d = np.full((9, 9), 10.0)
    seg = np.full((9, 9), 4, np.uint8)
    seg[4, 4] = 2  # a tree pixel
    ok = candidate_mask(d, seg, Sampling(10, 1, 1, 0.1, 3))
    assert not ok[3:6, 3:6].any()
    assert not ok[0].any() and ok[1, 1]


def test_flat_ground_oracle_and_relative_modes(flat_climate):
    r = evaluate_trajectory(flat_climate, "trajectory_0000", S, seed=0)
    assert len(r["frame"]) > 100
    # Oracle plane on exactly flat ground: only float16 depth quantisation remains.
    ok = r["oracle_plane.valid"]
    assert ok.all()
    assert np.max(np.abs(r["oracle_plane.rel_range_err"])) < 2e-3
    # Ground is flat at Down = 0 in every frame, so the relative plane is also exact.
    assert np.max(np.abs(r["relative_altitude.rel_range_err"])) < 2e-3
    np.testing.assert_allclose(r["z_oracle"], 0.0, atol=0.05)
    assert np.max(r["slope_deg"]) < 1.0


def test_relative_altitude_tracks_first_frame_reference(flat_climate):
    r = evaluate_trajectory(flat_climate, "trajectory_0000", S, seed=0)
    assert np.all(r["z_rel_ref"] == r["z_rel_ref"][0])
    traj = MidAirTrajectory(flat_climate, "trajectory_0000")
    assert traj.n_frames == 3
    traj.close()
