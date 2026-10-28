import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_midair as am  # noqa: E402


def test_move_verified_copies_then_deletes(tmp_path):
    src = tmp_path / "a" / "x.png"
    src.parent.mkdir()
    src.write_bytes(b"frame-data")
    dst = tmp_path / "b" / "x.png"
    am.move_verified(src, dst)
    assert not src.exists() and dst.read_bytes() == b"frame-data"


def test_move_verified_keeps_source_if_copy_differs(tmp_path, monkeypatch):
    src = tmp_path / "x.png"
    src.write_bytes(b"frame-data")
    dst = tmp_path / "out" / "x.png"
    real = am.sha256
    monkeypatch.setattr(am, "sha256", lambda p: "bad" if p == dst else real(p))
    with pytest.raises(RuntimeError):
        am.move_verified(src, dst)
    assert src.exists() and not dst.exists()


def test_trajectory_complete_needs_all_markers_and_no_partials(tmp_path):
    for m in am.REQUIRED:
        (tmp_path / m / "trajectory_0001").mkdir(parents=True)
        (tmp_path / m / "trajectory_0001" / ".frames.done").write_text("x")
    assert am.trajectory_complete(tmp_path, "trajectory_0001")
    (tmp_path / "depth" / "trajectory_0001" / "frames.zip.part").write_text("x")
    assert not am.trajectory_complete(tmp_path, "trajectory_0001")
    assert not am.trajectory_complete(tmp_path, "trajectory_0002")


def test_select_names_restricts_to_chosen_trajectories_and_refuses_strays():
    splits = {"development": ["t1", "t2", "t3"], "test": ["t9"]}
    assert am.select_names(splits, ["development"], None) == ["t1", "t2", "t3"]
    assert am.select_names(splits, ["development"], ["t3", "t1"]) == ["t1", "t3"]
    with pytest.raises(ValueError):
        am.select_names(splits, ["development"], ["t9"])


def _traj(root, name, modalities, n=3, size=5):
    for m in modalities:
        d = root / m / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i:06d}.png").write_bytes(b"x" * size)


def test_release_deletes_local_frames_only_when_drive_copies_match(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "drive"
    _traj(local, "t1", am.MODALITIES)
    _traj(remote, "t1", am.REQUIRED)
    counts = am.release_trajectory(local, remote, "t1")
    assert counts == {m: 3 for m in am.MODALITIES}
    for m in am.REQUIRED:
        assert not list((local / m / "t1").glob("*.png"))
        assert (local / m / "t1" / ".archived").exists()
    assert (local / "color_left" / "t1" / ".released").exists()


def test_release_refuses_and_deletes_nothing_if_a_drive_copy_differs(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "drive"
    _traj(local, "t1", am.MODALITIES)
    _traj(remote, "t1", am.REQUIRED)
    (remote / "normals" / "t1" / "000002.png").write_bytes(b"xy")
    with pytest.raises(RuntimeError):
        am.release_trajectory(local, remote, "t1")
    for m in am.MODALITIES:
        assert len(list((local / m / "t1").glob("*.png"))) == 3


def test_restore_verified_skips_matching_copy_and_retries_transient_errors(tmp_path, monkeypatch):
    src = tmp_path / "drive" / "x.png"
    src.parent.mkdir()
    src.write_bytes(b"frame-data")
    dst = tmp_path / "local" / "x.png"
    assert am.restore_verified(src, dst) is True
    assert am.restore_verified(src, dst) is False  # already there and identical
    dst.write_bytes(b"partial")  # an interrupted copy is replaced
    calls = {"n": 0}
    real_copy = am.shutil.copy2

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(121, "The semaphore timeout period has expired")
        return real_copy(a, b)

    monkeypatch.setattr(am.shutil, "copy2", flaky)
    assert am.restore_verified(src, dst, wait_s=0.0) is True
    assert dst.read_bytes() == b"frame-data" and calls["n"] == 2
