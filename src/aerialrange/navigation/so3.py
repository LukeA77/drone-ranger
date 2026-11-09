"""Rotation group helpers (SO(3)): skew matrix, exponential and logarithm maps.

Standard closed forms (Rodrigues' formula); see e.g. Sola, Deray and Atchuthan (2018),
"A micro Lie theory for state estimation in robotics", arXiv:1812.01537, Section 4 and
Appendix A.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def skew(v: ArrayLike) -> NDArray[np.float64]:
    """Skew-symmetric matrix [v]x so that [v]x w = v x w."""
    x, y, z = np.asarray(v, float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix for rotation vector `w` (radians)."""
    w = np.asarray(w, float).reshape(3)
    th = float(np.linalg.norm(w))
    W = skew(w)
    if th < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(th) / th * W + (1.0 - np.cos(th)) / th**2 * W @ W


def log_so3(R: ArrayLike) -> NDArray[np.float64]:
    """Rotation vector of rotation matrix `R` (inverse of `exp_so3`, angle in [0, pi])."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if np.pi - th < 1e-6:  # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-15))
        return th * axis / np.linalg.norm(axis)
    return th / (2.0 * np.sin(th)) * v
