"""Which frame are Mid-Air's IMU vectors in? (D-741)

Usage: python scripts/msckf_imu_frames.py

For each development flight:
- the ground-truth angular velocity is regressed on the rates derived from consecutive
  ground-truth attitudes, once as body rates log(R_k^T R_k+1)/dt and once as world rates
  log(R_k+1 R_k^T)/dt (3x3 matrix and residual standard deviation);
- the accelerometer is regressed on the ideal specific force in the body frame R^T (a - g)
  and in the world frame (a - g);
- 1 s open-loop integration of the noise-free ground-truth rates and accelerations from the
  true state, as body rates and as world rates, held per step and averaged per step.
Writes results/midair/development/navigation/imu_frames.json.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import h5py
import numpy as np

from aerialrange.geometry.frames import quat_to_R
from aerialrange.navigation.alignment import GRAVITY_NED
from aerialrange.navigation.so3 import exp_so3, log_so3
from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import midair_root, split_names  # noqa: E402

DT = 0.01


def regress(x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    A, *_ = np.linalg.lstsq(x, y, rcond=None)
    return A.T.round(4).tolist(), (y - x @ A).std(0).round(5).tolist()


def open_loop(
    R: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    frame: str,
    mean: bool,
) -> list[float]:
    errs = []
    for s0 in range(0, len(R) - 101, 300):
        Rk, pk, vk = R[s0], p[s0], v[s0]
        for k in range(s0, s0 + 100):
            wk = (w[k] + w[k + 1]) / 2 if mean else w[k]
            fk = (f[k] + f[k + 1]) / 2 if mean else f[k]
            E, H = exp_so3(wk * DT), exp_so3(wk * DT / 2)
            Rm = Rk @ H if frame == "body" else H @ Rk
            aw = Rm @ fk + GRAVITY_NED
            pk = pk + vk * DT + 0.5 * aw * DT**2
            vk = vk + aw * DT
            Rk = Rk @ E if frame == "body" else E @ Rk
        errs.append(np.degrees(np.linalg.norm(log_so3(Rk.T @ R[s0 + 100]))))
    return [float(np.median(errs)), float(np.max(errs))]


def main() -> int:
    out = {}
    with h5py.File(midair_root() / "sensor_records.hdf5", "r") as h:
        for name in split_names("development"):
            g = h[name]
            R = np.array([quat_to_R(q, order="wxyz") for q in g["groundtruth/attitude"][()]])
            w = g["groundtruth/angular_velocity"][()]
            a = g["groundtruth/acceleration"][()]
            p = g["groundtruth/position"][()]
            v = g["groundtruth/velocity"][()]
            acc = g["imu/accelerometer"][()]
            wb = np.array([log_so3(R[k].T @ R[k + 1]) / DT for k in range(len(R) - 1)])
            ww = np.array([log_so3(R[k + 1] @ R[k].T) / DT for k in range(len(R) - 1)])
            fb = np.einsum("nji,nj->ni", R, a - GRAVITY_NED)
            fw = a - GRAVITY_NED
            one = np.ones((len(acc), 1))
            out[name] = {
                "angvel_vs_body_rates": regress(w[:-1], wb),
                "angvel_vs_world_rates": regress(w[:-1], ww),
                "accel_vs_body_specific_force": regress(np.c_[fb, one], acc),
                "accel_vs_world_specific_force": regress(np.c_[fw, one], acc),
                "open_loop_1s_deg_median_max": {
                    f"{fr}_{'mean' if m else 'held'}": open_loop(R, p, v, w, fb, fr, m)
                    for fr in ("body", "world")
                    for m in (False, True)
                },
            }
            print(name, out[name]["open_loop_1s_deg_median_max"], flush=True)
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "note": "regress returns (matrix, residual sd); matrix maps the regressor to the target",
        "flights": out,
    }
    dest = REPO_ROOT / "results" / "midair" / "development" / "navigation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "imu_frames.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
