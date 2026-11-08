"""Motion-model settings of the IMM filter (D-728) from SynDrone development truth (D-735).

Usage: python scripts/syndrone_imm_fit.py

Reads the actors' true positions (cache/syndrone/speed/<town>.csv.gz, 25 Hz, position jitter
about 15 micrometres) of the development towns. Per group (people: CARLA walkers; vehicles:
cars, trucks, buses, vans, motorbikes; riders on bicycles are left out, as in scoring), every
sample is labelled with a motion mode and each model's noise is measured from the truth.

*Time scale.* CARLA's actors wobble from frame to frame (walker animation, vehicle physics;
acceleration correlation times of about one frame), which a filter whose measurements carry
decimetres of noise never sees. A white-noise model's density is therefore measured at the
time scale the filter works on, from increments over an interval T: for a random walk of
density q, var(x(t + T) - x(t)) = q T per axis. For velocity this estimate settles for
T >= 0.4 s (vehicles) and T >= 1 s (people) (D-735); the densities are the mean over
T = 1, 2 and 4 s, and the turn labels use the heading change over `T_LABEL` = 1 s.

- *Heading* is defined only from `V_HEADING` = 1 m/s: below it CARLA's creeping, shuffling and
  reversing actors give turn rates with a 90th percentile of 2 to 37 rad/s, while from 1 m/s it
  settles at 0.1 to 0.5 rad/s in every development town for both groups (D-735).
- *Labels:* stationary when the true speed is within 3 sigma of zero (sigma: the truth's own
  speed noise, scripts/syndrone_speed_truth.py); turning when the turn rate over 1 s exceeds
  3 sigma of its straight-motion noise (robust spread, 1.4826 x MAD, over samples with a
  heading; most motion is straight); otherwise constant velocity (including moving samples
  below 1 m/s).
- `q_cv`: velocity increments over intervals that are constant velocity throughout.
- `q_ct`: on intervals that are turning throughout, the velocity increment after the turn's
  own rotation of the velocity (by the mean turn rate over the interval) is removed.
- `q_turn`: increments of the 1 s turn rate over intervals with a heading throughout.
- `turn_rate_sigma0`: root mean square of the 1 s turn rate (a new track's turn is unknown).
- `velocity_sigma_moving`: root mean square velocity per axis over moving samples (constant
  velocity and turning), the spread a new track's velocity is judged against when its first
  modes are set.
- `switch_rates` (per second): transitions between labels divided by the time spent in the mode
  they leave; `mode0`: the share of time in each mode (a track starts at a random moment).
`q_stationary`, `gate_prob` and `reset_after` concern measurement and tracking errors, not true
motion, and are chosen on the development speed results (scripts/syndrone_speed_compare.py).
Output: results/syndrone/development/imm_fit.json.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aerialrange.data.syndrone import FPS, PERSON, VEHICLE_TAGS
from aerialrange.geometry.imm import MODE_NAMES, N_MODES
from aerialrange.utils.config import REPO_ROOT

DT = 1.0 / FPS
GROUP_TAGS = {"person": (PERSON,), "vehicle": VEHICLE_TAGS}
K_SIGMA = 3.0  # a value counts as non-zero beyond 3 sigma of its noise
V_HEADING = 1.0  # m/s, heading defined from this speed (see above)
T_LABEL = 1.0  # s, turn rate for labels: heading change over this interval
T_DENSITY = (1.0, 2.0, 4.0)  # s, intervals over which densities are measured (plateau)


def actor_pieces(df: pd.DataFrame, sig_v: float) -> list[dict]:
    """Per contiguous truth segment: velocity, speed, 1 s turn rate and stationarity."""
    half = int(round(T_LABEL * FPS / 2))
    out = []
    for (_, seg), g in df.groupby(["id", "segment"]):
        if seg < 0 or len(g) < 2 * half + 3:
            continue
        g = g.sort_values("frame")
        if (np.diff(g["frame"].to_numpy()) != 1).any():
            continue
        p = g[["x", "y"]].to_numpy(float)
        v = (p[2:] - p[:-2]) / (2 * DT)  # central differences
        sp = np.hypot(v[:, 0], v[:, 1])
        headed = sp >= V_HEADING
        hd = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        w = np.full(len(v), np.nan)  # turn rate over T_LABEL, centred
        ok = headed[2 * half :] & headed[: -2 * half]
        dh = (hd[2 * half :] - hd[: -2 * half]) / (2 * half * DT)
        w[half:-half] = np.where(ok, dh, np.nan)
        out.append({"v": v, "sp": sp, "w": w, "still": sp <= K_SIGMA * sig_v})
    return out


def label(pieces: list[dict]) -> float:
    """Set each piece's mode labels; returns the straight-motion noise of the 1 s turn rate."""
    ws = np.concatenate([p["w"][np.isfinite(p["w"])] for p in pieces])
    sig_w = float(1.4826 * np.median(np.abs(ws - np.median(ws))))
    for p in pieces:
        m = np.full(len(p["v"]), 1)  # constant velocity
        with np.errstate(invalid="ignore"):
            m[np.abs(p["w"]) > K_SIGMA * sig_w] = 2  # turning (NaN compares False)
        m[p["still"]] = 0  # stationary
        p["mode"] = m
    return sig_w


def all_true(mask: np.ndarray, lag: int) -> np.ndarray:
    """For each start i, whether mask[i : i + lag + 1] is all True (length n - lag)."""
    c = np.concatenate([[0], np.cumsum(~mask)])
    return (c[lag + 1 :] - c[: len(mask) - lag]) == 0


def increments(pieces: list[dict], key: str, mode: int | None, lag: int) -> np.ndarray:
    """Increments of `key` over `lag` samples on intervals wholly in `mode` (None: wherever
    the quantity is defined), stacked as rows. For velocity on turning intervals, the turn's
    own rotation of the velocity (mean turn rate over the interval) is removed first."""
    out = []
    for p in pieces:
        x = p[key] if p[key].ndim == 2 else p[key][:, None]
        if len(x) <= lag:
            continue
        ok = np.isfinite(x).all(axis=1)
        if mode is not None:
            ok &= p["mode"] == mode
        sel = all_true(ok, lag)
        if not sel.any():
            continue
        d = x[lag:] - x[:-lag]
        if key == "v" and mode == 2:
            wc = np.concatenate([[0.0], np.cumsum(np.nan_to_num(p["w"]))])
            wbar = (wc[lag + 1 :] - wc[: len(x) - lag]) / (lag + 1)
            ang = wbar * lag * DT
            c, s = np.cos(ang), np.sin(ang)
            v0 = x[:-lag]
            rot = np.column_stack([c * v0[:, 0] - s * v0[:, 1], s * v0[:, 0] + c * v0[:, 1]])
            d = x[lag:] - rot
        out.append(d[sel])
    return np.concatenate(out) if out else np.zeros((0, 1))


def density(pieces: list[dict], key: str, mode: int | None) -> dict:
    """Random-walk density of `key` per axis, var(increment) / T, for each T and their mean."""
    by_t = {}
    for t in T_DENSITY:
        d = increments(pieces, key, mode, int(round(t * FPS)))
        by_t[f"{t:g}"] = {
            "q": float(np.mean(d**2) / t) if len(d) else float("nan"),
            "n": int(len(d)),
        }
    qs = [v["q"] for v in by_t.values() if np.isfinite(v["q"])]
    return {"q": float(np.mean(qs)) if qs else float("nan"), "by_interval_s": by_t}


def fit_group(pieces: list[dict]) -> dict:
    sig_w = label(pieces)
    time_in = np.zeros(N_MODES)
    trans = np.zeros((N_MODES, N_MODES))
    for p in pieces:
        m = p["mode"]
        time_in += np.bincount(m, minlength=N_MODES) * DT
        np.add.at(trans, (m[:-1], m[1:]), 1)
    np.fill_diagonal(trans, 0)
    rates = np.divide(trans, time_in[:, None], out=np.zeros_like(trans), where=time_in[:, None] > 0)
    cv, ct, turn = density(pieces, "v", 1), density(pieces, "v", 2), density(pieces, "w", None)
    ws = np.concatenate([p["w"][np.isfinite(p["w"])] for p in pieces])
    vm = np.concatenate([p["v"][p["mode"] > 0] for p in pieces])
    return {
        "velocity_sigma_moving": float(np.sqrt(np.mean(vm**2))),
        "q_cv": cv["q"],
        "q_ct": ct["q"],
        "q_turn": turn["q"],
        "turn_rate_sigma0": float(np.sqrt(np.mean(ws**2))),
        "switch_rates": rates.tolist(),
        "mode0": (time_in / time_in.sum()).tolist(),
        "detail": {
            "straight_turn_rate_noise_rad_s": sig_w,
            "q_cv_by_interval": cv["by_interval_s"],
            "q_ct_by_interval": ct["by_interval_s"],
            "q_turn_by_interval": turn["by_interval_s"],
            "seconds_in_mode": dict(zip(MODE_NAMES, time_in.tolist(), strict=True)),
            "transitions": trans.tolist(),
            "mean_dwell_s": {
                MODE_NAMES[i]: float(time_in[i] / trans[i].sum()) if trans[i].sum() else None
                for i in range(N_MODES)
            },
        },
    }


def main() -> int:
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text("utf-8"))["syndrone"][
        "splits"
    ]
    out: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-735",
        "towns": splits["development"],
        "frame_rate_hz": FPS,
        "v_heading_ms": V_HEADING,
        "t_label_s": T_LABEL,
        "t_density_s": list(T_DENSITY),
        "groups": {},
    }
    by_group: dict[str, list[dict]] = {g: [] for g in GROUP_TAGS}
    sig_v = {}
    for town in splits["development"]:
        df = pd.read_csv(REPO_ROOT / "cache" / "syndrone" / "speed" / f"{town}.csv.gz")
        summ = json.loads(
            (REPO_ROOT / "results/syndrone/development" / f"speed_truth_{town}.json").read_text()
        )
        sig_v[town] = float(summ["speed_sigma_from_jitter_ms"])
        for g, tags in GROUP_TAGS.items():
            by_group[g] += actor_pieces(df[df["tag"].isin(tags)], sig_v[town])
        print(f"{town}: done", flush=True)
    out["truth_speed_sigma_ms"] = sig_v
    for g, pcs in by_group.items():
        out["groups"][g] = fit_group(pcs)
        out["groups"][g]["segments"] = len(pcs)
    dest = REPO_ROOT / "results" / "syndrone" / "development" / "imm_fit.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["groups"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
