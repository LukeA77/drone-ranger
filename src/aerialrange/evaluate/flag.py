"""Reliability flag (plan E): probability that an estimate is a "large error".

A small L2-regularised logistic regression on interpretable per-estimate features (for the
terrain method: calibrated relative sigma, local roughness, support radius relative to range,
parallax). Features are standardised with the training mean and spread. It is fitted on the
calibration split only and evaluated on the test split. Kept deliberately small
and transparent: the coefficients are reported, so a reviewer can see what drives the flag.
Fitting uses Newton-Raphson (iteratively reweighted least squares); the L2 weight is a
documented setting (`{{FLAG_L2}}`), not tuned on test data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class FlagModel:
    names: list[str]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    coef: NDArray[np.float64]
    intercept: float

    def score(self, X: ArrayLike) -> NDArray[np.float64]:
        """Probability of a large error for each row of features."""
        Z = (np.asarray(X, float) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(Z @ self.coef + self.intercept)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlagModel:
        return cls(
            list(d["names"]),
            np.asarray(d["mean"]),
            np.asarray(d["std"]),
            np.asarray(d["coef"]),
            float(d["intercept"]),
        )


def fit_flag(
    X: ArrayLike,
    y: ArrayLike,
    names: Sequence[str],
    l2: float,
    n_iter: int = 50,
    tol: float = 1e-10,
) -> FlagModel:
    """Fit by Newton-Raphson on the penalised log-likelihood (intercept not penalised)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if not np.all(np.isfinite(X)):
        raise ValueError("features must be finite; drop or impute rows first")
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], l2)
    pen[0] = 0.0
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (p - y) / len(y) + pen * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + np.diag(pen)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return FlagModel(list(names), mean, std, w[1:], float(w[0]))
