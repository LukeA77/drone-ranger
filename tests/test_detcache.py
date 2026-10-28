"""Saved detections replayed into the tracker (validation grid)."""

import numpy as np
import pytest

from aerialrange.track.detcache import Replay, load_sequence, save_sequence


def _dets():
    return [
        (np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]), np.array([0.9, 0.15]), np.array([3, 0])),
        (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)),
        (np.array([[9.0, 9, 9, 9]]), np.array([0.5]), np.array([4])),
    ]


def test_round_trip_keeps_frames_and_empty_frames(tmp_path):
    p = tmp_path / "seq.npz"
    save_sequence(p, _dets(), [(240, 320)] * 3)
    dets, shapes = load_sequence(p)
    assert [len(s) for _, s, _ in dets] == [2, 0, 1]
    np.testing.assert_allclose(dets[0][0], _dets()[0][0])
    assert dets[2][2].tolist() == [4] and shapes.tolist() == [[240, 320]] * 3


def test_replay_returns_frames_in_order_and_filters_low_scores(tmp_path):
    p = tmp_path / "seq.npz"
    save_sequence(p, _dets(), [(240, 320)] * 3)
    r = Replay(*load_sequence(p), min_conf=0.2)
    img = np.zeros((240, 320, 3), np.uint8)
    assert r(img)[1].tolist() == [0.9]
    assert len(r(img)[1]) == 0
    assert r(img)[2].tolist() == [4]
    with pytest.raises(IndexError):
        r(img)


def test_replay_refuses_frames_of_another_size(tmp_path):
    p = tmp_path / "seq.npz"
    save_sequence(p, _dets(), [(240, 320)] * 3)
    with pytest.raises(ValueError):
        Replay(*load_sequence(p))(np.zeros((100, 100, 3), np.uint8))


def test_replay_through_tracker_matches_live_detection(tmp_path):
    """Replaying saved detections gives the same tracks as calling the detector live."""
    pytest.importorskip("ultralytics")
    from aerialrange.track.runner import run_sequence_custom
    from aerialrange.utils.config import REPO_ROOT

    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(12)]
    live = [
        (np.array([[40.0 + 6 * k, 100, 20, 30]]), np.array([0.9]), np.array([3])) for k in range(12)
    ]
    cfg = str(REPO_ROOT / "configs" / "trackers" / "botsort_cmc.yaml")
    calls = iter(live)
    a, _ = run_sequence_custom(lambda img: next(calls), frames, cfg, {3: "vehicle"})
    save_sequence(tmp_path / "s.npz", live, [(240, 320)] * 12)
    replay = Replay(*load_sequence(tmp_path / "s.npz"))
    b, _ = run_sequence_custom(replay, frames, cfg, {3: "vehicle"})
    np.testing.assert_allclose(a["vehicle"], b["vehicle"])


def test_tracker_overrides_change_behaviour_and_refuse_unknown_keys():
    pytest.importorskip("ultralytics")
    from aerialrange.track.runner import make_tracker, run_sequence_custom
    from aerialrange.utils.config import REPO_ROOT

    cfg = str(REPO_ROOT / "configs" / "trackers" / "bytetrack.yaml")
    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(8)]

    def det(img):
        return np.array([[40.0, 100, 20, 30]]), np.array([0.5]), np.array([3])

    on, _ = run_sequence_custom(det, frames, cfg, {3: "vehicle"})
    off, _ = run_sequence_custom(det, frames, cfg, {3: "vehicle"}, {"new_track_thresh": 0.6})
    assert len(on["vehicle"]) > 0 and len(off["vehicle"]) == 0
    with pytest.raises(KeyError):
        make_tracker(cfg, {"no_such_key": 1})


def test_gmc_downscale_override():
    pytest.importorskip("ultralytics")
    from aerialrange.track.runner import make_tracker
    from aerialrange.utils.config import REPO_ROOT

    bot = str(REPO_ROOT / "configs" / "trackers" / "botsort_cmc.yaml")
    assert make_tracker(bot).gmc.downscale == 2  # library default
    assert make_tracker(bot, {"gmc_downscale": 4}).gmc.downscale == 4
    with pytest.raises(KeyError):
        make_tracker(
            str(REPO_ROOT / "configs" / "trackers" / "bytetrack.yaml"), {"gmc_downscale": 4}
        )


def test_replay_stays_in_step_under_a_detection_stride(tmp_path):
    pytest.importorskip("ultralytics")
    from aerialrange.track.runner import run_sequence_custom
    from aerialrange.utils.config import REPO_ROOT

    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(10)]
    live = [
        (np.array([[40.0 + 6 * k, 100, 20, 30]]), np.array([0.9]), np.array([3])) for k in range(10)
    ]
    save_sequence(tmp_path / "s.npz", live, [(240, 320)] * 10)
    replay = Replay(*load_sequence(tmp_path / "s.npz"))
    cfg = str(REPO_ROOT / "configs" / "trackers" / "bytetrack.yaml")
    rows, _ = run_sequence_custom(replay, frames, cfg, {3: "vehicle"}, det_stride=2)
    assert replay.k == 10  # every frame consumed, detected or skipped
    v = rows["vehicle"]
    detected = v[v[:, 0] == 9][0]  # frame 9 is a detection frame: box at 40 + 6 * 8
    assert detected[2] == pytest.approx(88.0, abs=2.0)
