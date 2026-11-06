"""Data-check tests on tiny SYNTHETIC fixtures (no real datasets)."""

import zipfile

import h5py
import pytest

from aerialrange.data.check import (
    check_auair,
    check_midair,
    check_visdrone_det,
    check_visdrone_mot,
    parse_midair_download_list,
)

MOT = "VisDrone2019-MOT-val"
GOOD_ROW = "1,1,10,20,30,40,1,4,0,0\n2,1,11,21,30,40,1,4,0,0\n1,0,0,0,5,5,0,0,0,0\n"


def _make_mot(base, rows=GOOD_ROW, n_frames=2, as_zip=False):
    root = base / MOT
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "seq1.txt").write_text(rows)
    (root / "sequences" / "seq1").mkdir(parents=True)
    for i in range(1, n_frames + 1):
        (root / "sequences" / "seq1" / f"{i:07d}.jpg").write_bytes(b"x")
    if as_zip:
        with zipfile.ZipFile(base / f"{MOT}.zip", "w") as z:
            for p in root.rglob("*"):
                z.write(p, p.relative_to(base).as_posix())
        for p in sorted(root.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        root.rmdir()


@pytest.mark.parametrize("as_zip", [False, True])
def test_visdrone_mot_ok(tmp_path, as_zip):
    _make_mot(tmp_path, as_zip=as_zip)
    r = check_visdrone_mot(tmp_path, MOT)
    assert r.status == "ok", r.problems
    assert r.details["sequences"] == 1
    assert r.details["frames"] == 2
    assert r.details["boxes"] == 3
    assert r.details["ignored_region_rows"] == 1


def test_visdrone_mot_missing(tmp_path):
    assert check_visdrone_mot(tmp_path, MOT).status == "missing"


def test_visdrone_mot_wrong_column_count_fails_loudly(tmp_path):
    _make_mot(tmp_path, rows="1,1,10,20,30,40,1,4,0\n")
    r = check_visdrone_mot(tmp_path, MOT)
    assert r.status == "schema_error"
    assert "9 columns" in r.problems[0]


def test_visdrone_mot_bad_category(tmp_path):
    _make_mot(tmp_path, rows="1,1,10,20,30,40,1,12,0,0\n")
    assert check_visdrone_mot(tmp_path, MOT).status == "schema_error"


def test_visdrone_mot_frame_index_beyond_frames(tmp_path):
    _make_mot(tmp_path, rows="5,1,10,20,30,40,1,4,0,0\n", n_frames=2)
    r = check_visdrone_mot(tmp_path, MOT)
    assert any("max frame index" in p for p in r.problems)


def test_visdrone_det_ok(tmp_path):
    root = tmp_path / "VisDrone2019-DET-val"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "images" / "a.jpg").write_bytes(b"x")
    (root / "annotations" / "a.txt").write_text("1,2,3,4,1,4,0,0,\n")
    r = check_visdrone_det(tmp_path, "VisDrone2019-DET-val")
    assert r.status == "ok", r.problems
    assert r.details["boxes"] == 1


DL = (
    "https://midair.ulg.ac.be/MidAir/Kite_training/sunny/sensor_records.zip?id=SECRET\n"
    "https://midair.ulg.ac.be/MidAir/Kite_training/sunny/depth/trajectory_0000/frames.zip?id=SECRET\n"
)


def test_midair_download_list_drops_token(tmp_path):
    (tmp_path / "dl.txt").write_text(DL)
    exp = parse_midair_download_list(tmp_path / "dl.txt")
    assert exp == {"Kite_training/sunny": {"sensor_records": set(), "depth": {"trajectory_0000"}}}
    assert "SECRET" not in repr(exp)


def test_midair_only_download_list_is_missing(tmp_path):
    (tmp_path / "download_config.txt").write_text(DL)
    [r] = check_midair(tmp_path)
    assert r.status == "missing"


def test_midair_complete(tmp_path):
    (tmp_path / "download_config.txt").write_text(DL)
    cdir = tmp_path / "Kite_training" / "sunny"
    (cdir / "depth" / "trajectory_0000").mkdir(parents=True)
    (cdir / "depth" / "trajectory_0000" / "000000.PNG").write_bytes(b"x")
    with h5py.File(cdir / "sensor_records.hdf5", "w") as f:
        for g in ("camera_data", "gps", "groundtruth", "imu"):
            f.create_group(f"trajectory_0000/{g}")
    [r] = check_midair(tmp_path)
    assert r.status == "ok", r.problems
    assert r.details["trajectories_in_hdf5"] == 1


def test_midair_unextracted_zip_is_partial(tmp_path):
    (tmp_path / "download_config.txt").write_text(DL)
    cdir = tmp_path / "Kite_training" / "sunny"
    cdir.mkdir(parents=True)
    (cdir / "sensor_records.zip").write_bytes(b"x")
    [r] = check_midair(tmp_path)
    assert r.status == "partial"
    assert any("not extracted" in p for p in r.problems)


def test_auair_unconfigured_is_missing():
    assert check_auair(None).status == "missing"


# --- SynDrone cache (synthetic index and members; no network) ---


def _syndrone_cache(tmp_path, n_frames=2, keep=None, drop=None):
    """A fake SynDrone cache: both archive indexes for Town01 and a few kept members."""
    import json
    import zlib

    from aerialrange.data.check import _expected_syndrone

    camera = json.dumps({"x": 1, "y": 2, "z": 50, "pitch": -90, "yaw": 0, "roll": 0}).encode()
    box = json.dumps([]).encode()
    base = tmp_path / "syndrone"
    data = {}
    for archive in ("color", "depth"):
        names = sorted(_expected_syndrone("Town01", archive, n_frames) - set(drop or ()))
        members = []
        for n in names:
            b = camera if "/camera/" in n else box if "/bboxes/" in n else n.encode()
            data[n] = b
            members.append([n, 0, len(b), len(b), zlib.crc32(b), 8])
        (base / "_index").mkdir(parents=True, exist_ok=True)
        idx = base / "_index" / f"Town01_Opt_120_{archive}.zip.index.json"
        idx.write_text(json.dumps({"url": "x", "size": 1, "members": members, "limit": {}}))
    for n in keep or [n for n in data if "/camera/" in n or "/bboxes/" in n]:
        (base / n).parent.mkdir(parents=True, exist_ok=True)
        (base / n).write_bytes(data[n])
    return base


def test_syndrone_cache_ok_and_unfetched_town_is_remote(tmp_path):
    from aerialrange.data.check import check_syndrone

    base = _syndrone_cache(tmp_path)
    r1, r2 = check_syndrone(base, ["Town01", "Town02"], n_frames=2)
    assert r1.status == "ok", r1.problems
    assert r1.details["crc_checked"] == 2 * 3 + 2  # camera at 3 heights and boxes, 2 frames
    assert r2.status == "remote"


def test_syndrone_corrupt_member_fails(tmp_path):
    from aerialrange.data.check import check_syndrone

    base = _syndrone_cache(tmp_path)
    f = base / "Town01_Opt_120/ClearNoon/height50m/camera/00001.json"
    f.write_bytes(f.read_bytes().replace(b"50", b"51"))
    (r,) = check_syndrone(base, ["Town01"], n_frames=2)
    assert r.status == "schema_error"
    assert any("CRC-32" in p and "height50m" in p for p in r.problems)


def test_syndrone_incomplete_archive_and_stray_files(tmp_path):
    from aerialrange.data.check import check_syndrone

    gone = "Town01_Opt_120/ClearNoon/height80m/depth/00001.png"
    base = _syndrone_cache(tmp_path, drop=[gone])
    (base / "Town01_Opt_120/ClearNoon/bboxes/00000.json.part").write_bytes(b"[")
    (r,) = check_syndrone(base, ["Town01"], n_frames=2)
    assert r.status == "schema_error"
    assert any("depth: 1 expected members not in the archive" in p for p in r.problems)
    assert any(".part" in p for p in r.problems)


def test_syndrone_member_that_does_not_parse(tmp_path):
    import json
    import zlib

    from aerialrange.data.check import check_syndrone

    base = _syndrone_cache(tmp_path)
    name = "Town01_Opt_120/ClearNoon/height20m/camera/00000.json"
    bad = b'{"x": 1}'
    idx = base / "_index" / "Town01_Opt_120_color.zip.index.json"
    d = json.loads(idx.read_text())
    for m in d["members"]:
        if m[0] == name:
            m[3], m[4] = len(bad), zlib.crc32(bad)
    idx.write_text(json.dumps(d))
    (base / name).write_bytes(bad)
    (r,) = check_syndrone(base, ["Town01"], n_frames=2)
    assert r.status == "schema_error"
    assert any("does not parse as camera" in p for p in r.problems)
