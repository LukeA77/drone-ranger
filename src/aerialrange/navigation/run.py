"""Run the filter over one recorded flight (D-741). Data-source agnostic: arrays and a frame
reader in, per-frame estimates and covariances out.

Order per camera frame k (IMU sample 4k at 100 Hz and 25 Hz): propagate the IMU samples since
frame k - 1; GPS update when a fix falls on this frame; clone the pose; track features; use
tracks that ended and, when the window is full, every track seen in the oldest clone (its
measurements are consumed, as in OpenVINS); marginalise the oldest clone.

Start: IMU white noise from the first `align_min_s` seconds (refreshed at every GPS fix from all
samples so far, causal); attitude from the in-flight alignment (`navigation.alignment`);
position and velocity from the GPS fix at the end of the alignment window; biases from the
dataset's stored initial estimates with the generator's stated error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aerialrange.navigation.alignment import VisionPair, align
from aerialrange.navigation.features import FeatureTracker, TrackerSettings
from aerialrange.navigation.gnss import GnssErrorModel
from aerialrange.navigation.heading import travel_direction
from aerialrange.navigation.imu import INIT_BIAS_EST_NOISE_FRACTION, ImuNoise
from aerialrange.navigation.msckf import BA, BG, GC, GM, NCORE, PO, TH, VE, Msckf, MsckfSettings

REL_GAPS = (1, 2, 5, 10)  # frames; version 1 terrain from motion's gaps (configs/geometry.yaml)


@dataclass(frozen=True)
class AlignSettings:
    min_window_s: float = 2.0  # OpenVINS init_window_time
    max_window_s: float = 10.0
    max_sigma_rad: float = float(np.deg2rad(5.0))


@dataclass
class FlightInput:
    gyr: NDArray  # (N, 3) rad/s, body
    acc: NDArray  # (N, 3) m/s^2, body (specific force)
    imu_hz: float
    gps_pos: NDArray  # (G, 3) m, world NED
    gps_vel: NDArray  # (G, 3) m/s, world NED
    gps_hz: float
    n_frames: int
    cam_hz: float
    read_gray: Callable[[int], NDArray]
    K: NDArray
    R_bc: NDArray
    t_bc: NDArray
    bg0: NDArray
    ba0: NDArray


@dataclass
class FlightOutput:
    frames: NDArray
    t: NDArray
    R: NDArray
    p: NDArray
    v: NDArray
    bg: NDArray
    ba: NDArray
    gps_offset: NDArray
    P_tpv: NDArray  # (n, 9, 9) covariance of (attitude, position, velocity) errors
    align_window_s: float
    align_sigma_deg: NDArray
    info: dict = field(default_factory=dict)
    # in-window pose pairs for triangulation (camera runs only): for each frame k and each gap
    # g in REL_GAPS, the current estimates of the clones at k and k - g and the 12x12 joint
    # covariance of (position k, attitude k, position k-g, attitude k-g); NaN when missing
    win_rot: NDArray | None = None  # (n, 1 + len(REL_GAPS), 3, 3): clone k, then k - g
    win_pos: NDArray | None = None  # (n, 1 + len(REL_GAPS), 3)
    win_cov: NDArray | None = None  # (n, len(REL_GAPS), 12, 12)


def vision_pairs(
    d: FlightInput, gnss: GnssErrorModel, ts: TrackerSettings, n_fix: int, per_gps_frames: int
) -> list[VisionPair]:
    """Camera travel direction from each GPS fix's frame to REL_GAPS[-1] frames later, with the
    GPS direction over the same time: displacement v_i dt + a_i dt^2 / 2, a_i the forward
    difference of GPS velocity. Its error: GPS velocity noise times dt and the acceleration
    term's uncertainty, taken as half the change between consecutive acceleration estimates."""
    gap = REL_GAPS[-1]
    dt = gap / d.cam_hz
    gdt = 1.0 / d.gps_hz
    rng = np.random.default_rng(0)
    out = []
    for i in range(min(n_fix, len(d.gps_vel) - 2)):
        k = i * per_gps_frames
        if k + gap >= d.n_frames:
            break
        td = travel_direction(d.read_gray(k), d.read_gray(k + gap), d.K, ts, rng)
        if td is None:
            continue
        a_i = (d.gps_vel[i + 1] - d.gps_vel[i]) / gdt
        a_n = (d.gps_vel[i + 2] - d.gps_vel[i + 1]) / gdt
        disp = d.gps_vel[i] * dt + 0.5 * a_i * dt**2
        dist = float(np.linalg.norm(disp))
        if dist <= 0:
            continue
        err = np.sqrt((gnss.vel_sigma * dt) ** 2 + (0.25 * np.linalg.norm(a_n - a_i) * dt**2) ** 2)
        out.append(
            VisionPair(
                i * gdt, dt, td.d_c, td.sigma_rad, disp / dist, float(np.sqrt(2) * err / dist)
            )
        )
    return out


def run_flight(
    d: FlightInput,
    gnss: GnssErrorModel,
    s: MsckfSettings,
    ts: TrackerSettings,
    al: AlignSettings,
    use_camera: bool = True,
    max_frames: int | None = None,
) -> FlightOutput:
    imu_dt = 1.0 / d.imu_hz
    per_frame = round(d.imu_hz / d.cam_hz)
    # step samples: mean of consecutive raw samples (the rate over each step; on Mid-Air's
    # noise-free motion this cuts the 1 s open-loop attitude error from 0.03 to 0.0004 deg)
    gyr_s = 0.5 * (d.gyr[:-1] + d.gyr[1:])
    acc_s = 0.5 * (d.acc[:-1] + d.acc[1:])
    per_gps_frames = round(d.cam_hz / d.gps_hz)
    n0 = round(al.min_window_s * d.imu_hz)
    noise = ImuNoise.from_signal(d.gyr[:n0], d.acc[:n0])
    # bias priors for the window mean: the stored estimate's error (n / 50, generator) and a
    # random walk over the longest window (variance of a walk's mean over N steps ~ N rw^2 / 3)
    f0 = INIT_BIAS_EST_NOISE_FRACTION
    n_max = al.max_window_s * d.imu_hz
    bg_sig = np.sqrt((f0 * noise.gyr) ** 2 + noise.gyr_bias_rw**2 * n_max / 3)
    ba_sig = np.sqrt((f0 * noise.acc) ** 2 + noise.acc_bias_rw**2 * n_max / 3)
    vision = (
        vision_pairs(d, gnss, ts, int(al.max_window_s * d.gps_hz), per_gps_frames)
        if use_camera
        else None
    )
    a = align(
        gyr_s,
        acc_s,
        imu_dt,
        d.gps_vel,
        1.0 / d.gps_hz,
        d.bg0,
        bg_sig,
        d.ba0,
        ba_sig,
        noise.acc,
        gnss.vel_sigma,
        al.min_window_s,
        al.max_window_s,
        al.max_sigma_rad,
        s.gyro_frame,
        noise.gyr,
        vision,
        d.R_bc,
    )
    g0 = round(a.window_s * d.gps_hz)  # GPS index at the start
    k0 = g0 * per_gps_frames  # first filter frame
    r0 = k0 * per_frame  # its IMU row
    noise = ImuNoise.from_signal(d.gyr[:r0], d.acc[:r0])
    R_start = a.R_end
    P0 = np.zeros((NCORE, NCORE))
    idx = np.r_[0:3, 9:12, 12:15]  # attitude, gyro bias, accel bias
    P0[np.ix_(idx, idx)] = a.cov_end
    # gyro white noise integrated over the window; bias at the window end vs its window mean
    P0[TH, TH] += np.diag(noise.gyr**2 * imu_dt**2 * r0)
    P0[BG, BG] += np.diag(noise.gyr_bias_rw**2 * r0 / 3)
    P0[BA, BA] += np.diag(noise.acc_bias_rw**2 * r0 / 3)
    c2, m2, w2 = gnss.const_sigma**2, gnss.gm_sigma**2, gnss.white_sigma**2
    P0[PO, PO] = np.diag(c2 + m2 + w2)
    P0[PO, GC] = P0[GC, PO] = -np.diag(c2)
    P0[PO, GM] = P0[GM, PO] = -np.diag(m2)
    P0[GC, GC] = np.diag(c2)
    P0[GM, GM] = np.diag(m2)
    P0[VE, VE] = np.eye(3) * gnss.vel_sigma**2
    flt = Msckf(
        d.R_bc,
        d.t_bc,
        float(d.K[0, 0]),
        s,
        noise,
        gnss,
        k0 / d.cam_hz,
        R_start,
        d.gps_pos[g0],
        d.gps_vel[g0],
        a.bg,
        a.ba,
        P0,
    )
    tracker = FeatureTracker(ts)
    tracks: dict[int, list[tuple[int, NDArray]]] = {}
    Kinv = np.linalg.inv(d.K)
    last = d.n_frames if max_frames is None else min(d.n_frames, k0 + max_frames)
    last = min(last, len(gyr_s) // per_frame)
    rec: dict[str, list] = {
        k: [] for k in ("k", "t", "R", "p", "v", "bg", "ba", "gc", "P", "wR", "wp", "wC")
    }
    used = tri_rej = chi_rej = tried = 0
    gps_nis: list[float] = []
    feat_nis: list[float] = []
    for k in range(k0, last):
        if k > k0:
            rows = slice((k - 1) * per_frame, k * per_frame)
            flt.propagate(gyr_s[rows], acc_s[rows], imu_dt)
        if s.gps_updates and k % per_gps_frames == 0 and k > k0:
            gi = k // per_gps_frames
            if gi < len(d.gps_pos):
                noise = ImuNoise.from_signal(d.gyr[: k * per_frame], d.acc[: k * per_frame])
                flt.noise = noise
                gps_nis.append(flt.update_gps(d.gps_pos[gi], d.gps_vel[gi]))
        if use_camera:
            flt.augment(k)
            ids, uv = tracker.step(d.read_gray(k))
            xy = (np.c_[uv, np.ones(len(uv))] @ Kinv.T)[:, :2]
            current = set(ids.tolist())
            for i, q in zip(ids.tolist(), xy, strict=True):
                tracks.setdefault(i, []).append((k, q))
            use = []
            for i in [i for i in tracks if i not in current]:
                if len(tracks[i]) >= s.min_obs:
                    use.append(tracks[i])
                del tracks[i]
            if len(flt.clones) > s.max_clones:
                old = flt.oldest_clone()
                for i, obs in tracks.items():
                    if obs and obs[0][0] == old:
                        if len(obs) >= s.min_obs:
                            use.append(obs)
                            tracks[i] = []
                        else:
                            tracks[i] = obs[1:]
            if use:
                st = flt.update_features(use)
                used += st.features_used
                tri_rej += st.rejected_triangulation
                chi_rej += st.rejected_chi2
                tried += st.features_tried
                feat_nis += st.nis
            if len(flt.clones) > s.max_clones:
                flt.marginalize(flt.oldest_clone())
        idx = np.r_[0:9]
        rec["k"].append(k)
        rec["t"].append(flt.t)
        rec["R"].append(flt.R.copy())
        rec["p"].append(flt.p.copy())
        rec["v"].append(flt.v.copy())
        rec["bg"].append(flt.bg.copy())
        rec["ba"].append(flt.ba.copy())
        rec["gc"].append(flt.gc + flt.gm)
        rec["P"].append(flt.P[np.ix_(idx, idx)].copy())
        if use_camera:
            wR = np.full((1 + len(REL_GAPS), 3, 3), np.nan)
            wp = np.full((1 + len(REL_GAPS), 3), np.nan)
            wC = np.full((len(REL_GAPS), 12, 12), np.nan)
            if k in flt.clones:
                ik = flt.clone_index(k)
                wR[0], wp[0] = flt.clones[k].R, flt.clones[k].p
                for j, g in enumerate(REL_GAPS):
                    if k - g in flt.clones:
                        ig = flt.clone_index(k - g)
                        wR[1 + j], wp[1 + j] = flt.clones[k - g].R, flt.clones[k - g].p
                        # (position, attitude) order; clone layout is (attitude, position)
                        sel = np.r_[ik + 3 : ik + 6, ik : ik + 3, ig + 3 : ig + 6, ig : ig + 3]
                        wC[j] = flt.P[np.ix_(sel, sel)]
            rec["wR"].append(wR)
            rec["wp"].append(wp)
            rec["wC"].append(wC)
    return FlightOutput(
        frames=np.array(rec["k"]),
        t=np.array(rec["t"]),
        R=np.array(rec["R"]),
        p=np.array(rec["p"]),
        v=np.array(rec["v"]),
        bg=np.array(rec["bg"]),
        ba=np.array(rec["ba"]),
        gps_offset=np.array(rec["gc"]),
        P_tpv=np.array(rec["P"]),
        win_rot=np.array(rec["wR"]) if use_camera else None,
        win_pos=np.array(rec["wp"]) if use_camera else None,
        win_cov=np.array(rec["wC"]) if use_camera else None,
        align_window_s=a.window_s,
        align_sigma_deg=np.rad2deg(np.sqrt(np.diag(a.cov_end[:3, :3]))),
        info={
            "features_used": used,
            "features_tried": tried,
            "rejected_triangulation": tri_rej,
            "rejected_chi2": chi_rej,
            "feature_nis_mean": float(np.mean(feat_nis)) if feat_nis else None,
            "gps_nis_mean": float(np.mean(gps_nis)) if gps_nis else None,
            "imu_noise_final": {
                "gyr": noise.gyr.tolist(),
                "acc": noise.acc.tolist(),
            },
        },
    )
