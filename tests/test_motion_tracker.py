"""The project's tracker (track/tracker.py) and its parts, on synthetic scenes with answers."""

from dataclasses import dataclass

import numpy as np
import pytest

from aerialrange.track.box_kalman import BoxKalman
from aerialrange.track.camera_motion import CameraMotion
from aerialrange.track.runner import coast, make_tracker
from aerialrange.track.tracker import MotionTracker, TrackerSettings, assign, iou_xyxy
from aerialrange.utils.config import REPO_ROOT


@dataclass
class Dets:
    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


def dets(boxes, conf=0.9, cls=0):
    b = np.asarray(boxes, float).reshape(-1, 4)
    n = len(b)
    return Dets(b, np.full(n, conf, float), np.full(n, cls, float))


BLANK = np.zeros((200, 300, 3), np.uint8)


# --- Kalman filter ----------------------------------------------------------------------


def test_kalman_learns_constant_velocity():
    kf = BoxKalman()
    m, c = kf.initiate(np.array([[50.0, 40.0, 20.0, 10.0]]))
    for k in range(1, 40):
        m, c = kf.predict(m, c)
        m, c = kf.update(m, c, np.array([[50.0 + 2 * k, 40.0 - k, 20.0, 10.0]]))
    np.testing.assert_allclose(m[0, 4:6], [2.0, -1.0], atol=0.05)
    np.testing.assert_allclose(m[0, 6:8], [0.0, 0.0], atol=0.05)
    assert np.all(np.linalg.eigvalsh(c[0]) > 0)


def test_kalman_warp_moves_centre_and_scales_size():
    kf = BoxKalman()
    m, c = kf.initiate(np.array([[10.0, 20.0, 4.0, 6.0]]))
    m[0, 4:6] = [1.0, 0.0]
    a = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, -1.0]])  # rotate 90 degrees, scale 2, shift
    m2, c2 = kf.warp(m, c, a)
    np.testing.assert_allclose(m2[0, :2], [0 * 10 - 2 * 20 + 5, 2 * 10 + 0 * 20 - 1])
    np.testing.assert_allclose(m2[0, 2:4], [8.0, 12.0])  # size scales, not rotates
    np.testing.assert_allclose(m2[0, 4:6], [0.0, 2.0])  # velocity rotates and scales
    m3, c3 = kf.warp(m, c, np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    np.testing.assert_allclose(m3, m)
    np.testing.assert_allclose(c3, c)


def test_lost_tracks_stop_changing_size():
    kf = BoxKalman()
    m, c = kf.initiate(np.array([[0.0, 0.0, 10.0, 10.0]]))
    m[0, 6:8] = 1.0
    m2, _ = kf.predict(m, c, freeze_size=np.array([True]))
    np.testing.assert_allclose(m2[0, 2:4], [10.0, 10.0])


# --- association ------------------------------------------------------------------------


def test_assign_finds_the_cheapest_matching_and_respects_the_gate():
    cost = np.array([[0.1, 0.2], [0.15, 0.9]])
    m, ur, uc = assign(cost, 0.8)
    assert sorted(m) == [(0, 1), (1, 0)]  # total 0.35, not 0.1 + 0.9
    m, ur, uc = assign(np.array([[0.85]]), 0.8)
    assert m == [] and ur == [0] and uc == [0]
    m, _, _ = assign(np.array([[0.1, np.inf]]), 0.8)
    assert m == [(0, 0)]


def test_iou_and_buffered_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [12, 0, 22, 10]])
    np.testing.assert_allclose(iou_xyxy(a, b)[0], [50 / 150, 0.0])
    assert iou_xyxy(a, b, buffer=0.2)[0, 1] > 0  # the gap of 2 px is bridged


# --- camera motion ----------------------------------------------------------------------


def textured(seed=0, shape=(240, 320)):
    rng = np.random.default_rng(seed)
    import cv2

    g = (rng.random((shape[0] // 4, shape[1] // 4)) * 255).astype(np.uint8)
    g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.GaussianBlur(g, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def shift(img, dx, dy):
    import cv2

    return cv2.warpAffine(
        img, np.float32([[1, 0, dx], [0, 1, dy]]), img.shape[1::-1], borderMode=cv2.BORDER_REFLECT
    )


@pytest.mark.parametrize("downscale", [1, 2])
def test_camera_motion_recovers_a_shift(downscale):
    base = textured()
    cm = CameraMotion(downscale)
    a0 = cm.estimate(base)
    np.testing.assert_allclose(a0, np.eye(2, 3))
    a = cm.estimate(shift(base, 6, -4), avoid_xyxy=np.array([[100, 100, 140, 140]]))
    np.testing.assert_allclose(a[:, :2], np.eye(2), atol=0.01)
    np.testing.assert_allclose(a[:, 2], [6, -4], atol=0.3)
    assert cm.n_inliers >= 10


def test_camera_motion_falls_back_to_identity_on_a_blank_frame():
    cm = CameraMotion()
    cm.estimate(BLANK)
    np.testing.assert_allclose(cm.estimate(BLANK), np.eye(2, 3))


# --- tracker ----------------------------------------------------------------------------

NO_CMC = TrackerSettings(high=0.5, low=0.1, new=0.6, cmc=False)


def run(tracker, frames):
    """frames: list of Dets or None (coast). Returns {frame: {id: box}}."""
    out = {}
    for k, d in enumerate(frames, start=1):
        if d is None:
            out[k] = {t.track_id: t.tlwh for t in tracker.coast(BLANK)}
        else:
            r = tracker.update(d, BLANK)
            out[k] = {int(x[4]): x[:4] for x in r}
    return out


def test_two_moving_objects_keep_their_identities():
    frames = [dets([[20 + 3 * k, 50, 16, 16], [200 - 3 * k, 60, 16, 16]]) for k in range(40)]
    out = run(MotionTracker(NO_CMC), frames)
    assert len(out[1]) == 2  # confirmed at once on the first frame
    ids = [sorted(out[k]) for k in range(1, 41)]
    assert all(i == ids[0] for i in ids)  # crossing paths at frame ~30 do not swap...
    first = {i: out[1][i][0] for i in ids[0]}
    last = {i: out[40][i][0] for i in ids[0]}
    assert sorted(first, key=first.get) == sorted(last, key=lambda i: -last[i])  # ...ids


def test_a_briefly_missed_object_keeps_its_id_and_a_long_gap_does_not():
    move = [dets([[20 + 2 * k, 50, 16, 16]]) for k in range(60)]
    gap = move[:10] + [dets([])] * 8 + move[18:]
    out = run(MotionTracker(NO_CMC), gap)
    assert set(out[5]) == set(out[25])
    s = TrackerSettings(**{**NO_CMC.__dict__, "max_lost": 5})
    out = run(MotionTracker(s), gap)
    assert set(out[5]) != set(out[25])


def test_a_single_detection_is_tentative_and_low_scores_do_not_start_tracks():
    t = MotionTracker(NO_CMC)
    t.update(dets([]), BLANK)  # frame 1 empty
    assert len(t.update(dets([[50, 50, 10, 10]]), BLANK)) == 0  # tentative
    assert len(t.update(dets([[51, 50, 10, 10]]), BLANK)) == 1  # confirmed
    assert len(t.update(dets([[150, 150, 10, 10]], conf=0.55), BLANK)) == 0
    t2 = MotionTracker(NO_CMC)
    for _ in range(3):
        assert len(t2.update(dets([[50, 50, 10, 10]], conf=0.3), BLANK)) == 0


def test_per_class_thresholds_start_and_match_tracks_by_class():
    s = TrackerSettings(**{**NO_CMC.__dict__, "high_by_class": {0: 0.2}, "new_by_class": {0: 0.25}})
    np.testing.assert_allclose(s.thresholds(np.array([0, 3]))[0], [0.2, NO_CMC.high])
    np.testing.assert_allclose(s.thresholds(np.array([0, 3]))[1], [0.25, NO_CMC.new])
    for cls, n_out in ((0, 1), (3, 0)):  # a 0.4 person starts a track; a 0.4 vehicle does not
        t = MotionTracker(s)
        t.update(dets([]), BLANK)
        t.update(dets([[50, 50, 10, 10]], conf=0.4, cls=cls), BLANK)
        assert len(t.update(dets([[50, 50, 10, 10]], conf=0.4, cls=cls), BLANK)) == n_out
    m = TrackerSettings.from_mapping({"high_by_class": {"1": 0.1}, "new_by_class": None})
    assert m.high_by_class == {1: 0.1} and m.new_by_class is None


def test_weak_detections_extend_a_track_in_stage_two():
    t = MotionTracker(NO_CMC)
    for k in range(5):
        t.update(dets([[50 + k, 50, 20, 20]]), BLANK)
    r = t.update(dets([[55, 50, 20, 20]], conf=0.2), BLANK)
    assert len(r) == 1 and r[0, 5] == pytest.approx(0.2)


def test_class_groups_stop_a_person_track_taking_a_vehicle_detection():
    s = TrackerSettings(**{**NO_CMC.__dict__, "class_groups": ((0, 1), (3, 4))})
    t = MotionTracker(s)
    for _ in range(3):
        t.update(dets([[50, 50, 20, 20]], cls=0), BLANK)
    r = t.update(dets([[50, 50, 20, 20]], cls=3), BLANK)
    assert len(r) == 0  # not matched (and the vehicle starts only a tentative track)
    t = MotionTracker(NO_CMC)
    for _ in range(3):
        t.update(dets([[50, 50, 20, 20]], cls=0), BLANK)
    r = t.update(dets([[50, 50, 20, 20]], cls=1), BLANK)
    assert len(r) == 1 and r[0, 6] == 1  # without groups the class follows the detection


def test_camera_motion_compensation_keeps_an_object_through_a_gap():
    base = textured(1, (240, 320))
    pan = 7  # px per frame: the camera pans, the object is fixed on the ground

    def frame(k):
        return shift(base, -pan * k, 0)

    def det(k):
        return dets([[200 - pan * k, 120, 12, 12]])

    seq = [det(k) if not 4 <= k < 10 else dets([]) for k in range(14)]
    results = {}
    for cmc in (False, True):
        s = TrackerSettings(high=0.5, low=0.1, new=0.6, cmc=cmc, cmc_downscale=1)
        t = MotionTracker(s)
        ids = []
        for k, d in enumerate(seq):
            r = t.update(d, frame(k))
            ids.append({int(x[4]) for x in r})
        results[cmc] = ids
    # with compensation the object is re-found under its first id; without it, the track
    # started at rest drifts nowhere while the object moves 42 px, and a new id begins
    assert results[True][12] == results[True][1]
    assert results[False][12] != results[False][1]


def test_coast_predicts_without_changing_states():
    t = MotionTracker(NO_CMC)
    for k in range(20):
        t.update(dets([[50 + 4 * k, 50, 20, 20]]), BLANK)
    out = coast(t, BLANK)
    assert len(out) == 1
    assert out[0].tlwh[0] + 10 == pytest.approx(50 + 4 * 20, abs=0.5)  # moved on by one step
    assert len(t.update(dets([[50 + 4 * 21, 50, 20, 20]]), BLANK)) == 1


def test_make_tracker_builds_the_project_tracker_from_yaml():
    cfg = str(REPO_ROOT / "configs" / "trackers" / "motion_parity.yaml")
    t = make_tracker(cfg, {"gmc_downscale": 8, "new": 0.6})
    assert isinstance(t, MotionTracker)
    assert t.s.cmc_downscale == 8 and t.s.new == 0.6 and t.s.high == 0.05
    # Library key names map to the project's settings, so one grid tunes both trackers.
    t = make_tracker(cfg, {"track_high_thresh": 0.3, "new_track_thresh": 0.45})
    assert t.s.high == 0.3 and t.s.new == 0.45
    with pytest.raises(KeyError):
        make_tracker(cfg, {"match_thresh_typo": 0.3})


def test_box_mask_matches_a_plain_loop():
    from aerialrange.track.camera_motion import box_mask

    rng = np.random.default_rng(3)
    xy = rng.uniform(-20, 330, (40, 2))
    boxes = np.column_stack([xy, xy + rng.uniform(1, 60, (40, 2))])
    for d in (1, 4):
        h, w = 240 // d, 320 // d
        ref = np.full((h, w), 255, np.uint8)
        for x0, y0, x1, y1 in boxes / d:
            ys = slice(max(0, int(np.floor(y0))), min(h, int(np.ceil(y1)) + 1))
            xs = slice(max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)) + 1))
            ref[ys, xs] = 0
        np.testing.assert_array_equal(box_mask((h, w), boxes, d), ref)


def test_assign_reduced_solve_equals_the_full_problem():
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(5)
    for _ in range(200):
        n, m = rng.integers(0, 9, 2)
        cost = rng.uniform(0, 1.5, (n, m))
        cost[rng.random((n, m)) < 0.3] = np.inf
        got, ur, uc = assign(cost, 0.8)
        # reference: the full augmented problem, no reduction
        big = 1e6
        c = np.where(np.isfinite(cost) & (cost <= 0.8), cost, big)
        aug = np.full((n + m, n + m), big)
        aug[:n, :m] = c
        aug[np.arange(n), m + np.arange(n)] = 0.4
        aug[n + np.arange(m), np.arange(m)] = 0.4
        aug[n:, m:] = 0
        if n + m:
            r, k = linear_sum_assignment(aug)
            ref_total = aug[r, k].sum()
        else:
            ref_total = 0.0
        total = sum(cost[a, b] for a, b in got) + 0.4 * (len(ur) + len(uc))
        assert total == pytest.approx(ref_total)
        assert all(cost[a, b] <= 0.8 for a, b in got)
        assert len(ur) == n - len(got) and len(uc) == m - len(got)
