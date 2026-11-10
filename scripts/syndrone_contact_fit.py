"""Fit the contact-point correction for a run's detector (D-743).

Usage: python scripts/syndrone_contact_fit.py --run v2_dfine [--group vehicle] [--boot 1000]

Rows: the run's matched vehicle track-frames on the development towns
(cache/syndrone/runs/<run>/range_matches_development.csv.gz, written by
scripts/syndrone_evaluate.py), with the footprint correction of D-736 on and no contact
correction. Model, written before any corrected result (D-743): the signed relative range
error against the footprint centre is a (shift_px) times 1 / (f tan theta) plus b (raise_m)
times cos^2 theta / H, the first-order effect of moving the ray down a pixels and of meeting
it with a plane b metres above the ground (f focal length in pixels, theta the truth's
depression, H camera height); no intercept; least absolute deviations (iteratively
reweighted); 95 % intervals by resampling whole sequences (town and height).
Writes results/syndrone/development/contact_fit_<run>.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aerialrange.data.syndrone import FOV_DEG, WIDTH
from aerialrange.utils.config import REPO_ROOT

RUNS = REPO_ROOT / "cache" / "syndrone" / "runs"
FOCAL_PX = WIDTH / 2 / np.tan(np.deg2rad(FOV_DEG) / 2)


def lad(X: np.ndarray, y: np.ndarray, iters: int = 500) -> np.ndarray:
    """Least absolute deviations by iteratively reweighted least squares."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(y - X @ b), 1e-6)
        bn = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        if np.max(np.abs(bn - b)) < 1e-10:
            return bn
        b = bn
    return b


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(df["depression_deg"].to_numpy(float))
    h = df["height"].to_numpy(float)
    X = np.c_[1.0 / (FOCAL_PX * np.tan(th)), np.cos(th) ** 2 / h]
    y = ((df["pred_range_m"] - df["truth_range_m"]) / df["truth_range_m"]).to_numpy(float)
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--group", default="vehicle")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(RUNS / args.run / "range_matches_development.csv.gz")
    df = df[df["group"] == args.group]
    X, y = design(df)
    a, b = lad(X, y)
    seqs = [g for _, g in df.groupby(["town", "height"])]
    parts = [design(g) for g in seqs]
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.boot):
        pick = rng.integers(0, len(parts), len(parts))
        Xb = np.vstack([parts[i][0] for i in pick])
        yb = np.concatenate([parts[i][1] for i in pick])
        boots.append(lad(Xb, yb))
    boots = np.array(boots)
    res = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "D-743",
        "run": args.run,
        "group": args.group,
        "rows": len(df),
        "sequences": len(seqs),
        "focal_px": FOCAL_PX,
        "shift_px": float(a),
        "raise_m": float(b),
        "shift_px_ci95": np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
        "raise_m_ci95": np.percentile(boots[:, 1], [2.5, 97.5]).tolist(),
        "config_value": {args.group: {"shift_px": round(a, 2), "raise_m": round(b, 3)}},
    }
    out = REPO_ROOT / "results" / "syndrone" / "development" / f"contact_fit_{args.run}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
