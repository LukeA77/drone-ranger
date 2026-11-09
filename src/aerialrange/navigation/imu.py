"""IMU noise model and its estimate from the IMU signal itself (D-741).

Mid-Air's IMU model (Fonder and Van Droogenbroeck, 2019, Section 2.1, and the dataset's
`tools/IMU-data_generator.py`): per sample and axis, measurement = truth + white noise
N(0, n) + bias, and the bias takes a Gaussian random-walk step N(0, b) after every sample.
n and b are drawn per flight and not recorded. The generator's bounds (per sample):
accelerometer n = 2 * 10^-U(1, 3) m/s^2, b = 2 * 10^-U(3, 6) m/s^2; gyroscope
n = 10^-U(1, 3) rad/s, b = 10^-U(4, 6) rad/s.

White noise n is estimated per axis from the signal: the true motion is smooth at 100 Hz, so
the second difference of the samples is dominated by noise, Var(d2 m) = 6 n^2, and a median
absolute deviation makes it robust to manoeuvres. Checked on all 15 development flights
against ground-truth residuals: ratio 0.98 to 1.05 over a flight, 0.82 to 1.22 from the first
3 s (scripts/msckf_noise_check.py). The bias random walk b cannot be seen this way; the
generator's upper bound is used (conservative: the filter then never trusts the bias too much).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Upper bounds of the Mid-Air generator's per-sample bias random-walk steps (see above).
ACC_BIAS_RW_MAX = 2e-3  # m/s^2 per sample
GYR_BIAS_RW_MAX = 1e-4  # rad/s per sample
# The generator's stored initial-bias estimate carries an error N(0, n / 50) per axis.
INIT_BIAS_EST_NOISE_FRACTION = 1.0 / 50.0


def white_noise_sigma(samples: ArrayLike) -> NDArray[np.float64]:
    """Per-axis white-noise standard deviation from the second difference (robust).

    `samples` is (N, 3) at a fixed rate. Returns (3,), in the samples' units.
    """
    x = np.asarray(samples, float)
    if len(x) < 5:
        raise ValueError("need at least 5 samples")
    d = np.diff(x, 2, axis=0)
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    return 1.4826 * mad / np.sqrt(6.0)


@dataclass(frozen=True)
class ImuNoise:
    """Per-axis, per-sample standard deviations (body frame)."""

    gyr: NDArray[np.float64]  # white noise, rad/s
    acc: NDArray[np.float64]  # white noise, m/s^2
    gyr_bias_rw: NDArray[np.float64]  # bias step per sample, rad/s
    acc_bias_rw: NDArray[np.float64]  # bias step per sample, m/s^2

    @classmethod
    def from_signal(cls, gyr: ArrayLike, acc: ArrayLike) -> ImuNoise:
        """White noise estimated from the samples; bias random walk at the generator's bound."""
        return cls(
            white_noise_sigma(gyr),
            white_noise_sigma(acc),
            np.full(3, GYR_BIAS_RW_MAX),
            np.full(3, ACC_BIAS_RW_MAX),
        )
