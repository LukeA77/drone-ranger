"""Fit the filter's GPS error model on the Mid-Air development flights (D-741).

Usage: python scripts/msckf_gnss_fit.py

Error e(t) = GPS position - ground-truth position at each 1 Hz fix, per axis (NED). Model:
random constant per flight + first-order Gauss-Markov part + white noise
(`aerialrange.navigation.gnss`). Fit:
- const_sigma: root mean square of the flight means (mean included);
- within a flight, the mean autocovariance of the de-meaned error at lags 1 to 10 s is fitted
  by sigma_gm^2 exp(-lag / tau) (one tau for all axes, least squares on a log-spaced tau grid),
  and white^2 = lag-0 variance - sigma_gm^2.
Also compared: the same statistics after dividing by the dilution of precision (HDOP for north
and east, VDOP for down); kept in metres if that does not reduce the spread between flights.
Also checks that GPS velocity equals ground truth (it does: no receiver noise in Mid-Air).
Writes results/midair/development/navigation/gnss_fit.json.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import h5py
import numpy as np

from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import midair_root, split_names  # noqa: E402

LAGS = np.arange(1, 11)


def errors(f: h5py.File, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    g = f[name]
    gp = g["gps/position"][()]
    p = g["groundtruth/position"][()]
    n = min(len(gp), (len(p) - 1) // 100 + 1)
    rows = np.arange(n) * 100
    e = gp[:n] - p[rows]
    dop = np.c_[
        g["gps/HDOP"][()].ravel()[:n], g["gps/HDOP"][()].ravel()[:n], g["gps/VDOP"][()].ravel()[:n]
    ]
    dv = g["gps/velocity"][()][:n] - g["groundtruth/velocity"][()][rows]
    return e, dop, float(np.abs(dv).max())


def autocov(es: list[np.ndarray]) -> np.ndarray:
    """Mean autocovariance (lags 0..10) of de-meaned errors, per axis: (11, 3)."""
    out = []
    for e in es:
        x = e - e.mean(0)
        out.append([np.mean(x * x, 0)] + [np.mean(x[:-k] * x[k:], 0) for k in LAGS])
    return np.mean(out, 0)


def fit_gm(ac: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best = None
    for tau in np.geomspace(0.3, 100.0, 400):
        basis = np.exp(-LAGS / tau)
        s2 = np.clip(ac[1:].T @ basis / (basis @ basis), 0, None)  # per axis
        sse = float(((ac[1:] - np.outer(basis, s2)) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, s2)
    _, tau, s2 = best
    s2 = np.minimum(s2, ac[0])
    return float(tau), np.sqrt(s2), np.sqrt(ac[0] - s2)


def main() -> int:
    root = midair_root()
    names = split_names("development")
    with h5py.File(root / "sensor_records.hdf5", "r") as f:
        data = [errors(f, n) for n in names]
    es = [e for e, _, _ in data]
    variants = {}
    for label, scale in (("metres", False), ("per_dop", True)):
        xs = [e / d if scale else e for e, d, _ in data]
        means = np.array([x.mean(0) for x in xs])
        sds = np.array([x.std(0) for x in xs])
        variants[label] = {
            "flight_mean_rms": np.sqrt((means**2).mean(0)).tolist(),
            "flight_mean_mean": means.mean(0).tolist(),
            "within_sd_coefficient_of_variation": (sds.std(0) / sds.mean(0)).tolist(),
        }
    cv_m = np.mean(variants["metres"]["within_sd_coefficient_of_variation"])
    cv_d = np.mean(variants["per_dop"]["within_sd_coefficient_of_variation"])
    chosen = "per_dop" if cv_d < cv_m else "metres"
    ac = autocov(es)
    tau, gm, white = fit_gm(ac)
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "split": "development",
        "flights": names,
        "variants": variants,
        "chosen_units": chosen,
        "autocovariance_m2_lags_0_to_10s": ac.tolist(),
        "gps_velocity_max_abs_diff_from_truth_ms": max(d for _, _, d in data),
        "model": {
            "const_sigma_m": variants["metres"]["flight_mean_rms"],
            "gm_sigma_m": gm.tolist(),
            "gm_tau_s": tau,
            "white_sigma_m": white.tolist(),
        },
    }
    if chosen != "metres":
        print("DOP scaling reduces the spread: the filter model needs per-DOP units (not built)")
    dest = REPO_ROOT / "results" / "midair" / "development" / "navigation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "gnss_fit.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["model"], indent=1), "\nchosen units:", chosen, cv_m, cv_d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
