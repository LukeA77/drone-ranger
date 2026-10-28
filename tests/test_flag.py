"""Reliability flag model (plan E): L2-regularised logistic regression, SYNTHETIC data."""

import numpy as np
import pytest

from aerialrange.evaluate.flag import FlagModel, fit_flag
from aerialrange.evaluate.stats import auroc


def synth(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.0 * X[:, 2] - 0.5
    y = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    return X, y


def test_recovers_signs_and_separates_classes():
    X, y = synth(5000, 0)
    m = fit_flag(X, y, names=["a", "b", "c"], l2=1e-3)
    w = dict(zip(m.names, m.coef, strict=True))
    assert w["a"] > 1.0 and w["b"] < -0.5 and abs(w["c"]) < 0.2
    Xt, yt = synth(5000, 1)
    assert auroc(m.score(Xt), yt) > 0.8


def test_standardisation_is_learned_on_training_data_only():
    X, y = synth(1000, 2)
    m = fit_flag(X * 100 + 5, y, names=["a", "b", "c"], l2=1e-3)
    assert np.allclose(m.mean, (X * 100 + 5).mean(axis=0))
    s1 = m.score(X[:5] * 100 + 5)
    assert np.all((s1 > 0) & (s1 < 1))


def test_deterministic_and_serialisable():
    X, y = synth(800, 3)
    a = fit_flag(X, y, names=["a", "b", "c"], l2=1e-2)
    b = fit_flag(X, y, names=["a", "b", "c"], l2=1e-2)
    np.testing.assert_array_equal(a.coef, b.coef)
    c = FlagModel.from_dict(a.to_dict())
    np.testing.assert_allclose(c.score(X), a.score(X))


def test_rejects_non_finite_features():
    X, y = synth(100, 4)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_flag(X, y, names=["a", "b", "c"], l2=1e-2)
