"""When to show a track's speed on the overlay.

A speed is shown only when the filter is confident: its speed sigma is at most
`sigma_max_ms` (fitted on SynDrone development towns), the track has
had at least `min_age` updates, and the mean normalised innovation squared (NIS) over its
last `nis_window` updates lies inside the two-sided `nis_conf` chi-square band for a
consistent filter (Bar-Shalom, Li and Kirubarajan, 2001, Section 5.4: the sum of N
two-dimensional NIS values is chi-square with 2N degrees of freedom). With `nis_test="upper"`
(D-739) only the upper bound applies, at the one-sided `nis_conf` quantile: the test then
catches an overconfident filter and does not reject a conservative one, whose interval widths
the calibration corrects (D-737). Otherwise the label
reads "speed: settling". Speed, heading and their sigmas are always written to tracks.csv;
this rule only affects what is displayed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from scipy.stats import chi2


@lru_cache(maxsize=64)
def nis_bounds(n: int, conf: float) -> tuple[float, float]:
    """Bounds on the mean of `n` two-dimensional NIS values for a consistent filter."""
    a = (1.0 - conf) / 2.0
    dof = 2 * n
    return float(chi2.ppf(a, dof) / n), float(chi2.ppf(1.0 - a, dof) / n)


@lru_cache(maxsize=64)
def nis_upper(n: int, conf: float) -> float:
    """One-sided upper bound on the mean of `n` two-dimensional NIS values."""
    return float(chi2.ppf(conf, 2 * n) / n)


@dataclass(frozen=True)
class SpeedDisplayRule:
    sigma_max_ms: float
    min_age: int
    nis_window: int
    nis_conf: float
    nis_test: str = "two_sided"  # or "upper" (D-739)

    def ok(self, speed_sigma_ms: float, age: int, nis_mean: float, nis_n: int) -> bool:
        if not (math.isfinite(speed_sigma_ms) and math.isfinite(nis_mean)):
            return False
        if speed_sigma_ms > self.sigma_max_ms or age < self.min_age:
            return False
        if nis_n < self.nis_window:
            return False
        if self.nis_test == "upper":
            return nis_mean <= nis_upper(self.nis_window, self.nis_conf)
        lo, hi = nis_bounds(self.nis_window, self.nis_conf)
        return lo <= nis_mean <= hi


def format_speed_label(
    rule: SpeedDisplayRule, speed_ms: float, sigma_ms: float, age: int, nis_mean: float, nis_n: int
) -> str:
    """'18 ± 3 km/h' when the rule allows it, else 'speed: settling'."""
    if not rule.ok(sigma_ms, age, nis_mean, nis_n):
        return "speed: settling"
    return f"{speed_ms * 3.6:.0f} \u00b1 {sigma_ms * 3.6:.0f} km/h"
