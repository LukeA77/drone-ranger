"""GPS error model for the filter (D-741).

Mid-Air simulates a single-frequency receiver from satellite pseudoranges, without the
ionosphere correction (Fonder and Van Droogenbroeck, 2019, Section 2.2), so its position error
is not white. Measured on the 15 development flights (scripts/msckf_gnss_fit.py), per axis:
a constant offset per flight (about 12 m too high, under 1 m horizontally), a part that decays
over a few seconds, and white noise. The model has three parts, per axis (north, east, down):
- `const_sigma`: random constant per flight, prior N(0, const_sigma^2), with const_sigma the
  root mean square of the flight means (mean included, so no dataset-specific mean is removed);
- first-order Gauss-Markov part with `gm_sigma` and time constant `gm_tau_s`;
- white noise `white_sigma`.
Scaling by the recorded dilution of precision was tested and made the spread between flights
larger, so the model is in metres (D-741).

Mid-Air's GPS velocity equals the ground-truth velocity (largest difference 7 mm/s on the
development flights): it carries no receiver noise. A filter fed it would get a velocity no real
receiver gives, so white noise is added with `vel_sigma` from a published receiver specification:
u-blox NEO-M8 data sheet (UBX-13003366, R10), velocity accuracy 0.05 m/s (50 % at 30 m/s). Read
as the median absolute error per axis of a zero-mean Gaussian, sigma = 0.05 / 0.6745 = 0.074 m/s
(the larger of the two readings; as a 3D median it would be 0.033 m/s). The added noise is
synthetic, seeded, and labelled as such in every result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

UBLOX_M8_VEL_ACC_50 = 0.05  # m/s, UBX-13003366 R10, "50 % @ 30 m/s"
GAUSS_MEDIAN_ABS = 0.6744897501960817  # median of |N(0, 1)|
VEL_SIGMA_DATASHEET = UBLOX_M8_VEL_ACC_50 / GAUSS_MEDIAN_ABS


@dataclass(frozen=True)
class GnssErrorModel:
    const_sigma: NDArray[np.float64]  # (3,) m
    gm_sigma: NDArray[np.float64]  # (3,) m
    gm_tau_s: float
    white_sigma: NDArray[np.float64]  # (3,) m
    vel_sigma: float = VEL_SIGMA_DATASHEET  # m/s per axis

    @classmethod
    def from_fit(cls, path: Path) -> GnssErrorModel:
        """Load the development fit written by scripts/msckf_gnss_fit.py."""
        d = json.loads(Path(path).read_text())["model"]
        return cls(
            np.asarray(d["const_sigma_m"], float),
            np.asarray(d["gm_sigma_m"], float),
            float(d["gm_tau_s"]),
            np.asarray(d["white_sigma_m"], float),
            float(d.get("vel_sigma_ms", VEL_SIGMA_DATASHEET)),
        )
