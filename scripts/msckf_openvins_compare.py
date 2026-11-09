"""Compare our filter with the OpenVINS reference on Mid-Air flights (Phase 11, D-741).

Usage: python scripts/msckf_openvins_compare.py --nav msckf_sig4
       --ov-dir D:/drone-ranger-data/openvins
       [--split development]

OpenVINS (camera and IMU only, no GPS; tools/openvins/) has its own world frame with an
arbitrary yaw and origin, so only measures that need no alignment are compared, on the frames
both have an estimate for:
- relative pose error at 1, 2, 5 and 10 frames, as in scripts/msckf_midair.py;
- tilt error: the angle between the estimated and the true gravity direction in the body frame
  (OpenVINS's world z is up; Mid-Air's is down).
OpenVINS's trajectory (ov_eval pose_to_file: time, position, quaternion x y z w, IMU to world)
is read from <ov-dir>/<flight>/openvins_estimate.txt; bag times are frame time + 1 s.
Writes results/midair/<split>/navigation/openvins_compare_<nav>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from aerialrange.geometry.frames import quat_to_R
from aerialrange.navigation.so3 import log_so3
from aerialrange.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import midair_root, split_names  # noqa: E402

GAPS = (1, 2, 5, 10)
NAV = REPO_ROOT / "cache" / "midair" / "navigation"
BAG_OFFSET_S = 1.0  # tools/openvins/make_bag.py stamps


def read_openvins(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    a = np.array([[float(x) for x in r[:8]] for r in rows])
    frames = np.round((a[:, 0] - BAG_OFFSET_S) * 25.0).astype(int)
    R = Rotation.from_quat(a[:, 4:8]).as_matrix()  # scipy order x y z w
    return frames, R, a[:, 1:4]


def rpe(frames: np.ndarray, R: np.ndarray, p: np.ndarray, Rt: dict, pt: dict) -> dict:
    idx = {int(k): i for i, k in enumerate(frames)}
    out = {}
    for g in GAPS:
        tr, ro = [], []
        for k, i in idx.items():
            j = idx.get(k + g)
            if j is None or k not in Rt or k + g not in Rt:
                continue
            dRt = Rt[k].T @ Rt[k + g]
            dpt = Rt[k].T @ (pt[k + g] - pt[k])
            dRe = R[i].T @ R[j]
            dpe = R[i].T @ (p[j] - p[i])
            tr.append(np.linalg.norm(dpe - dpt))
            ro.append(np.degrees(np.linalg.norm(log_so3(dRt.T @ dRe))))
        out[str(g)] = {
            "n": len(tr),
            "trans_median_m": float(np.median(tr)) if tr else None,
            "trans_p90_m": float(np.quantile(tr, 0.9)) if tr else None,
            "rot_median_deg": float(np.median(ro)) if ro else None,
        }
    return out


def tilt(frames: np.ndarray, R: np.ndarray, Rt: dict, down_world: np.ndarray) -> float:
    errs = []
    for k, Rk in zip(frames, R, strict=True):
        if int(k) not in Rt:
            continue
        g_est = Rk.T @ down_world
        g_true = Rt[int(k)].T @ np.array([0.0, 0.0, 1.0])
        errs.append(np.degrees(np.arccos(np.clip(g_est @ g_true, -1, 1))))
    return float(np.median(errs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", required=True)
    ap.add_argument("--ov-dir", required=True)
    ap.add_argument("--split", default="development")
    args = ap.parse_args(argv)
    res: dict = {}
    with h5py.File(midair_root() / "sensor_records.hdf5", "r") as h:
        for name in split_names(args.split):
            f_ov = Path(args.ov_dir) / name / "openvins_estimate.txt"
            f_us = NAV / args.nav / f"{name}.npz"
            if not f_ov.exists() or not f_us.exists():
                continue
            g = h[name]
            q = g["groundtruth/attitude"][()]
            pos = g["groundtruth/position"][()]
            fo, Ro, po = read_openvins(f_ov)
            us = np.load(f_us)
            common = sorted(set(map(int, fo)) & set(map(int, us["frames"])))
            if len(common) < 50:
                res[name] = {"error": f"only {len(common)} common frames"}
                continue
            Rt = {k: quat_to_R(q[4 * k], order="wxyz") for k in common}
            pt = {k: pos[4 * k] for k in common}
            mo = np.isin(fo, common)
            mu = np.isin(us["frames"], common)
            res[name] = {
                "common_frames": len(common),
                "openvins_first_frame": int(fo.min()),
                "ours": {
                    "rpe": rpe(us["frames"][mu], us["R"][mu], us["p"][mu], Rt, pt),
                    "tilt_err_median_deg": tilt(
                        us["frames"][mu], us["R"][mu], Rt, np.r_[0, 0, 1.0]
                    ),
                },
                "openvins": {
                    "rpe": rpe(fo[mo], Ro[mo], po[mo], Rt, pt),
                    "tilt_err_median_deg": tilt(fo[mo], Ro[mo], Rt, np.r_[0, 0, -1.0]),
                },
            }
            print(name, json.dumps(res[name])[:300], flush=True)
    ok = [v for v in res.values() if "ours" in v]
    pooled = (
        {
            who: {
                "tilt_err_median_deg": float(
                    np.median([v[who]["tilt_err_median_deg"] for v in ok])
                ),
                "rpe": {
                    g: {
                        k: float(np.median([v[who]["rpe"][g][k] for v in ok]))
                        for k in ("trans_median_m", "rot_median_deg")
                    }
                    for g in map(str, GAPS)
                },
            }
            for who in ("ours", "openvins")
        }
        if ok
        else {}
    )
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "split": args.split,
        "nav_run": args.nav,
        "note": "OpenVINS: camera and IMU only (no GPS), gyroscope rotated to body rates with "
        "the true attitude (tools/openvins/make_bag.py); ours: camera, IMU and GPS",
        "pooled_median_over_flights": pooled,
        "flights": res,
    }
    dest = REPO_ROOT / "results" / "midair" / args.split / "navigation"
    (dest / f"openvins_compare_{args.nav}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
