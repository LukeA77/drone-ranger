"""Run and score the camera, IMU and GPS filter on Mid-Air flights (Phase 11, D-741).

Usage: python scripts/msckf_midair.py --split development [--name msckf] [--no-camera]
       [--no-fej] [--sigma-px 1.0] [--workers 4] [--max-frames N] [--flights a,b]
       python scripts/msckf_midair.py --split test --final ...   (once, after freezing)

Per flight: GPS velocity gets synthetic white noise at the receiver data-sheet level
(`navigation.gnss`, seed = trajectory number), the filter runs from the end of the in-flight
alignment, and estimates are scored against ground truth at every frame:
- absolute trajectory error (no alignment: the GPS fixes the world frame), per axis and 3D;
- relative pose error at 1, 2, 5 and 10 frames (0.04 to 0.4 s, the baselines terrain from motion
  triangulates over): translation (m) and rotation (deg) of T_k^-1 T_k+d against the truth;
- attitude error split into tilt (roll and pitch: what ranging is most sensitive to) and yaw;
- NEES of attitude, position, velocity and all nine, against their chi-squared expectations.
Estimates are saved to cache/midair/navigation/<name>/<flight>.npz for the ranging comparison.
Writes results/midair/<split>/navigation/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.stats import chi2

from aerialrange.data.midair import MidAirTrajectory, midair_camera
from aerialrange.geometry.frames import quat_to_R, ypr_from_R_wb
from aerialrange.navigation.features import TrackerSettings
from aerialrange.navigation.gnss import GnssErrorModel
from aerialrange.navigation.msckf import MsckfSettings
from aerialrange.navigation.run import AlignSettings, FlightInput, FlightOutput, run_flight
from aerialrange.navigation.so3 import log_so3
from aerialrange.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from midair_common import midair_root, split_names  # noqa: E402

GAPS = (1, 2, 5, 10)
NAV_CFG = load_config("navigation")
FIT = REPO_ROOT / NAV_CFG.require("gnss_fit")
CACHE = REPO_ROOT / "cache" / "midair" / "navigation"


def load_input(root: Path, name: str, gnss: GnssErrorModel) -> tuple[FlightInput, dict]:
    traj = MidAirTrajectory(root, name)
    with h5py.File(root / "sensor_records.hdf5", "r") as f:
        g = f[name]
        gyr = g["imu/gyroscope"][()]
        acc = g["imu/accelerometer"][()]
        bg0 = g["imu/gyroscope"].attrs["init_bias_est"].ravel()
        ba0 = g["imu/accelerometer"].attrs["init_bias_est"].ravel()
        gps_pos = g["gps/position"][()]
        gps_vel_true = g["gps/velocity"][()]
        truth = {
            "p": g["groundtruth/position"][()],
            "v": g["groundtruth/velocity"][()],
            "q": g["groundtruth/attitude"][()],
        }
    rng = np.random.default_rng(int(name.split("_")[-1]))  # synthetic receiver noise, seeded
    gps_vel = gps_vel_true + rng.normal(size=gps_vel_true.shape) * gnss.vel_sigma
    first = cv2.imread(str(traj.frame_path("color_left", 0)), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape
    cam = midair_camera(w, h)
    paths = [str(traj.frame_path("color_left", k)) for k in range(traj.n_frames)]
    traj.close()

    def read_gray(k: int) -> np.ndarray:
        return cv2.imread(paths[k], cv2.IMREAD_GRAYSCALE)

    d = FlightInput(
        gyr=gyr,
        acc=acc,
        imu_hz=100.0,
        gps_pos=gps_pos,
        gps_vel=gps_vel,
        gps_hz=1.0,
        n_frames=len(paths),
        cam_hz=25.0,
        read_gray=read_gray,
        K=cam.K,
        R_bc=cam.R_bc,
        t_bc=cam.t_bc,
        bg0=bg0,
        ba0=ba0,
    )
    return d, truth


def score(out: FlightOutput, truth: dict) -> dict:
    rows = out.frames * 4
    Rt = np.array([quat_to_R(q, order="wxyz") for q in truth["q"][rows]])
    pt, vt = truth["p"][rows], truth["v"][rows]
    eth = np.array([log_so3(a @ b.T) for a, b in zip(Rt, out.R, strict=True)])
    ep, ev = pt - out.p, vt - out.v
    e9 = np.c_[eth, ep, ev]
    P = out.P_tpv
    nees = {}
    for label, sl in (
        ("attitude", slice(0, 3)),
        ("position", slice(3, 6)),
        ("velocity", slice(6, 9)),
        ("all9", slice(0, 9)),
    ):
        e = e9[:, sl]
        Pi = P[:, sl, sl]
        x = np.einsum("ni,ni->n", e, np.linalg.solve(Pi, e[..., None])[..., 0])
        dof = e.shape[1]
        nees[label] = {
            "mean": float(x.mean()),
            "expected": dof,
            "share_above_95": float(np.mean(x > chi2.ppf(0.95, dof))),
        }
    ypr_t = np.array([ypr_from_R_wb(R) for R in Rt])
    ypr_e = np.array([ypr_from_R_wb(R) for R in out.R])
    dyaw = np.rad2deg(np.angle(np.exp(1j * (ypr_e[:, 0] - ypr_t[:, 0]))))
    tilt = np.rad2deg(np.linalg.norm(eth[:, :2], axis=1))  # world-frame error about N and E
    rpe = {}
    for gap in GAPS:
        tr, ro = [], []
        for i in range(len(rows) - gap):
            j = i + gap
            dRt = Rt[i].T @ Rt[j]
            dRe = out.R[i].T @ out.R[j]
            dpt = Rt[i].T @ (pt[j] - pt[i])
            dpe = out.R[i].T @ (out.p[j] - out.p[i])
            tr.append(np.linalg.norm(dpe - dpt))
            ro.append(np.rad2deg(np.linalg.norm(log_so3(dRt.T @ dRe))))
        tr, ro = np.array(tr), np.array(ro)
        rpe[str(gap)] = {
            "seconds": gap / 25.0,
            "trans_median_m": float(np.median(tr)),
            "trans_p90_m": float(np.quantile(tr, 0.9)),
            "trans_rmse_m": float(np.sqrt(np.mean(tr**2))),
            "rot_median_deg": float(np.median(ro)),
            "rot_p90_deg": float(np.quantile(ro, 0.9)),
        }
    return {
        "frames": int(len(rows)),
        "start_s": float(out.t[0]),
        "align_window_s": out.align_window_s,
        "align_sigma_deg": out.align_sigma_deg.tolist(),
        "ate_rmse_m": float(np.sqrt(np.mean(np.sum(ep**2, 1)))),
        "ate_rmse_per_axis_m": np.sqrt(np.mean(ep**2, 0)).tolist(),
        "ate_horizontal_rmse_m": float(np.sqrt(np.mean(np.sum(ep[:, :2] ** 2, 1)))),
        "vel_rmse_ms": float(np.sqrt(np.mean(np.sum(ev**2, 1)))),
        "tilt_err_median_deg": float(np.median(tilt)),
        "tilt_err_p90_deg": float(np.quantile(tilt, 0.9)),
        "yaw_err_median_abs_deg": float(np.median(np.abs(dyaw))),
        "yaw_err_p90_abs_deg": float(np.quantile(np.abs(dyaw), 0.9)),
        "nees": nees,
        "rpe": rpe,
        "info": out.info,
    }


def run_one(args: tuple) -> tuple[str, dict]:
    name, name_out, s, ts, al, use_camera, max_frames = args
    cv2.setNumThreads(1)
    gnss = GnssErrorModel.from_fit(FIT)
    d, truth = load_input(midair_root(), name, gnss)
    out = run_flight(d, gnss, s, ts, al, use_camera=use_camera, max_frames=max_frames)
    dest = CACHE / name_out
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / f"{name}.npz",
        frames=out.frames,
        t=out.t,
        R=out.R,
        p=out.p,
        v=out.v,
        bg=out.bg,
        ba=out.ba,
        P_tpv=out.P_tpv,
        **(
            {"win_rot": out.win_rot, "win_pos": out.win_pos, "win_cov": out.win_cov}
            if out.win_rot is not None
            else {}
        ),
    )
    return name, score(out, truth)


def pooled(per: dict) -> dict:
    vals = list(per.values())
    out = {
        "ate_rmse_m_median": float(np.median([v["ate_rmse_m"] for v in vals])),
        "ate_horizontal_rmse_m_median": float(
            np.median([v["ate_horizontal_rmse_m"] for v in vals])
        ),
        "vel_rmse_ms_median": float(np.median([v["vel_rmse_ms"] for v in vals])),
        "tilt_err_median_deg_median": float(np.median([v["tilt_err_median_deg"] for v in vals])),
        "yaw_err_median_abs_deg_median": float(
            np.median([v["yaw_err_median_abs_deg"] for v in vals])
        ),
        "nees_mean": {
            k: float(np.mean([v["nees"][k]["mean"] for v in vals]))
            for k in ("attitude", "position", "velocity", "all9")
        },
        "rpe": {
            g: {
                k: float(np.median([v["rpe"][g][k] for v in vals]))
                for k in ("trans_median_m", "trans_p90_m", "rot_median_deg", "rot_p90_deg")
            }
            for g in map(str, GAPS)
        },
    }
    n = len(vals)
    for k, dof in (("attitude", 3), ("position", 3), ("velocity", 3), ("all9", 9)):
        lo, hi = chi2.ppf([0.025, 0.975], dof * n) / n
        out["nees_mean"][f"{k}_band_95"] = [float(lo), float(hi)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="development")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--name", default="msckf")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-fej", action="store_true")
    ap.add_argument("--no-gps-velocity", action="store_true")
    ap.add_argument("--no-gps-updates", action="store_true", help="GPS for the start only")
    ap.add_argument("--sigma-px", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--flights", default=None)
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        print("The test split runs once, after all settings are frozen. Use --final.")
        return 1
    s = MsckfSettings(**NAV_CFG.require("msckf"))  # configs/navigation.yaml (D-741)
    if args.no_fej:
        s = replace(s, fej=False)
    if args.no_gps_updates:
        s = replace(s, gps_updates=False)
    if args.no_gps_velocity:
        s = replace(s, use_gps_velocity=False)
    if args.sigma_px is not None:
        s = replace(s, sigma_px=args.sigma_px)
    ts = TrackerSettings(**NAV_CFG.require("tracker"))
    a = NAV_CFG.require("alignment")
    al = AlignSettings(a["min_window_s"], a["max_window_s"], float(np.deg2rad(a["max_sigma_deg"])))
    names = split_names(args.split)
    if args.flights:
        names = [n for n in names if n in args.flights.split(",")]
    jobs = [(n, args.name, s, ts, al, not args.no_camera, args.max_frames) for n in names]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        per = dict(ex.map(run_one, jobs))
    per = {n: per[n] for n in names}
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-741",
        "split": args.split,
        "name": args.name,
        "camera": not args.no_camera,
        "settings": asdict(s),
        "tracker": asdict(ts),
        "alignment": asdict(al),
        "gnss_model": json.loads(FIT.read_text())["model"],
        "gps_velocity_noise": "synthetic white noise at the u-blox NEO-M8 data-sheet level, "
        "seed = trajectory number (Mid-Air's GPS velocity is noise-free)",
        "pooled": pooled(per),
        "flights": per,
    }
    dest = REPO_ROOT / "results" / "midair" / args.split / "navigation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{args.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["pooled"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
