"""Check the IMU white-noise estimate from the signal against ground truth (D-741).

Usage: python scripts/msckf_noise_check.py

For each development flight and axis: the "truth" white-noise level is the first-difference
standard deviation / sqrt(2) of (IMU - ideal IMU from ground truth); the estimate is
`navigation.imu.white_noise_sigma` on the raw IMU samples (no ground truth), over the whole
flight and over the first 3 s. Writes results/midair/development/navigation/imu_noise_check.json.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import h5py
import numpy as np

from aerialrange.geometry.frames import quat_to_R
from aerialrange.navigation.alignment import GRAVITY_NED
from aerialrange.navigation.imu import white_noise_sigma
from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import midair_root, split_names  # noqa: E402


def main() -> int:
    names = split_names("development")
    rows = {}
    with h5py.File(midair_root() / "sensor_records.hdf5", "r") as f:
        for n in names:
            g = f[n]
            R = np.array([quat_to_R(q, order="wxyz") for q in g["groundtruth/attitude"][()]])
            a = g["groundtruth/acceleration"][()]
            w = g["groundtruth/angular_velocity"][()]
            acc = g["imu/accelerometer"][()]
            gyr = g["imu/gyroscope"][()]
            ideal_acc = np.einsum("nji,nj->ni", R, a - GRAVITY_NED)
            truth = np.r_[
                np.diff(gyr - w, axis=0).std(0), np.diff(acc - ideal_acc, axis=0).std(0)
            ] / np.sqrt(2)
            full = np.r_[white_noise_sigma(gyr), white_noise_sigma(acc)]
            first = np.r_[white_noise_sigma(gyr[:300]), white_noise_sigma(acc[:300])]
            rows[n] = {
                "truth_gyr_acc": truth.tolist(),
                "ratio_full_flight": (full / truth).tolist(),
                "ratio_first_3s": (first / truth).tolist(),
            }
    rf = np.array([r["ratio_full_flight"] for r in rows.values()])
    r3 = np.array([r["ratio_first_3s"] for r in rows.values()])
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "axes": ["gyr_x", "gyr_y", "gyr_z", "acc_x", "acc_y", "acc_z"],
        "flights": rows,
        "ratio_full_flight_min_max": [float(rf.min()), float(rf.max())],
        "ratio_first_3s_min_max": [float(r3.min()), float(r3.max())],
        "truth_range_gyr_rad_s": [float(rf.min()), float(rf.max())],
    }
    t = np.array([r["truth_gyr_acc"] for r in rows.values()])
    res["truth_range_gyr_rad_s"] = [float(t[:, :3].min()), float(t[:, :3].max())]
    res["truth_range_acc_m_s2"] = [float(t[:, 3:].min()), float(t[:, 3:].max())]
    dest = REPO_ROOT / "results" / "midair" / "development" / "navigation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "imu_noise_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: v for k, v in res.items() if k != "flights"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
