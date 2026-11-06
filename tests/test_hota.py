"""HOTA (vendored from TrackEval) on cases with hand-computed answers."""

import numpy as np
import pytest

from aerialrange.track.hota import hota_combine, hota_sequence, summary


def _seq(gt, tr, iou=1.0):
    """gt, tr: per-frame lists of ids; every gt/tr pair with the same position overlaps."""
    g = [np.array(x, int) for x in gt]
    t = [np.array(x, int) for x in tr]
    sims = [
        np.full((len(a), len(b)), iou) * np.eye(len(a), len(b)) for a, b in zip(g, t, strict=True)
    ]
    return g, t, sims


def test_perfect_tracking_is_one():
    r = hota_sequence(*_seq([[0, 1]] * 10, [[0, 1]] * 10))
    s = summary(r)
    assert s["HOTA"] == pytest.approx(1.0) and s["AssA"] == pytest.approx(1.0)


def test_no_predictions_is_zero():
    r = hota_sequence(*_seq([[0]] * 5, [[]] * 5))
    assert summary(r)["HOTA"] == 0.0 and r["HOTA_FN"][0] == 5


def test_identity_switch_halves_association():
    # One object, 10 frames; the tracker switches id halfway: AssA = 0.5 for every alpha.
    r = hota_sequence(*_seq([[0]] * 10, [[0]] * 5 + [[1]] * 5))
    assert np.allclose(r["DetA"], 1.0)
    assert np.allclose(r["AssA"], 0.5)
    assert summary(r)["HOTA"] == pytest.approx(np.sqrt(0.5))


def test_localisation_threshold():
    # IoU 0.62 everywhere: matched for alpha <= 0.6, missed (FN and FP) above.
    r = hota_sequence(*_seq([[0]] * 4, [[0]] * 4, iou=0.62))
    alphas = np.arange(0.05, 0.99, 0.05)
    assert np.allclose(r["DetA"][alphas <= 0.6 + 1e-9], 1.0)
    assert np.allclose(r["DetA"][alphas > 0.62], 0.0)
    assert summary(r)["HOTA"] == pytest.approx(np.mean(alphas <= 0.6 + 1e-9))


def test_combine_weights_by_true_positives():
    a = hota_sequence(*_seq([[0]] * 10, [[0]] * 10))
    b = hota_sequence(*_seq([[0]] * 10, [[0]] * 5 + [[1]] * 5))
    c = hota_combine({"a": a, "b": b})
    assert np.allclose(c["AssA"], 0.75) and np.allclose(c["DetA"], 1.0)
