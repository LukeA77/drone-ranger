"""Run Mid-Air ranging evaluation on one split.

Usage: python scripts/midair_eval.py --split development

Per-sample arrays go to cache/midair/<trajectory>.npz (git-ignored, regenerable).
A summary goes to results/midair/<split>/accuracy.json. Only fully downloaded trajectories
are used, and the summary lists which ones. The test split refuses to run without
--final, so it cannot be used during development.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np

from aerialrange.evaluate.midair_ranging import Sampling, evaluate_trajectory, sampling_dict
from aerialrange.utils.config import REPO_ROOT, load_config

MODES = ("oracle_plane", "relative_altitude")
CACHE = REPO_ROOT / "cache" / "midair"


def summarise(r: dict[str, np.ndarray], mode: str) -> dict[str, float]:
    """Point summaries of one mode's errors over valid samples."""
    ok = r[f"{mode}.valid"]
    if not ok.any():
        # A mode can be invalid for a whole trajectory (e.g. relative altitude when the drone
        # flies below its take-off height); report it rather than fail (D-607).
        keys = (
            "signed_range_err_median_m",
            "abs_range_err_median_m",
            "abs_range_err_mean_m",
            "range_rmse_m",
            "abs_range_err_p90_m",
            "abs_range_err_p95_m",
            "rel_range_err_median",
            "rel_range_err_p90",
            "horiz_err_median_m",
            "horiz_err_p95_m",
        )
        return {"n_samples": int(len(ok)), "valid_fraction": 0.0, **{k: float("nan") for k in keys}}
    e = r[f"{mode}.range_err_m"][ok]
    rel = np.abs(r[f"{mode}.rel_range_err"][ok])
    hz = r[f"{mode}.horiz_err_m"][ok]
    a = np.abs(e)
    return {
        "n_samples": int(len(ok)),
        "valid_fraction": float(ok.mean()),
        "signed_range_err_median_m": float(np.median(e)),
        "abs_range_err_median_m": float(np.median(a)),
        "abs_range_err_mean_m": float(a.mean()),
        "range_rmse_m": float(np.sqrt(np.mean(e**2))),
        "abs_range_err_p90_m": float(np.percentile(a, 90)),
        "abs_range_err_p95_m": float(np.percentile(a, 95)),
        "rel_range_err_median": float(np.median(rel)),
        "rel_range_err_p90": float(np.percentile(rel, 90)),
        "horiz_err_median_m": float(np.median(hz)),
        "horiz_err_p95_m": float(np.percentile(hz, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["development", "calibration", "test"])
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="summarise every trajectory of the split from cache/midair/<name>.npz without "
        "reading images (after batch processing; fails if any trajectory is missing)",
    )
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        print("The test split runs once, after all settings are frozen. Use --final.")
        return 1

    data = load_config("data")
    seed = load_config("project").require("seed")
    climate = data.require("midair_climate")
    root = REPO_ROOT / data.require("data_root") / data.require("midair_subdir") / climate
    s = Sampling(
        *(
            data.require(k)
            for k in (
                "pix_per_frame",
                "frame_stride",
                "edge_margin_px",
                "depth_edge_rel",
                "normal_patch_px",
            )
        )
    )
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["midair"]["splits"]
    if args.from_cache:
        absent = [n for n in splits[args.split] if not (CACHE / f"{n}.npz").exists()]
        if absent:
            print(f"--from-cache: no evaluation cache for {absent}")
            return 1
    names = (
        list(splits[args.split])
        if args.from_cache
        else [
            n
            for n in splits[args.split]
            if all(
                (root / m / n / ".frames.done").exists()
                for m in ("depth", "segmentation", "normals")
            )
            and not (root / "depth" / n / ".archived").exists()
        ]
    )
    archived = sorted(n for n in splits[args.split] if (root / "depth" / n / ".archived").exists())
    if archived:
        print(
            f"{len(archived)} trajectories are on Google Drive; restore them first with "
            f"scripts/archive_midair.py --restore --splits {args.split}: {archived}"
        )
    missing = sorted(set(splits[args.split]) - set(names))
    print(f"{args.split}: {len(names)} trajectories available, {len(missing)} not yet downloaded")

    CACHE.mkdir(parents=True, exist_ok=True)
    per_traj, all_r = {}, []
    for name in names:
        print(f"  {name} ...", flush=True)
        if args.from_cache:
            with np.load(CACHE / f"{name}.npz") as z:
                r = {k: z[k] for k in z.files}
        else:
            r = evaluate_trajectory(root, name, s, seed)
            np.savez_compressed(CACHE / f"{name}.npz", **r)
        per_traj[name] = {m: summarise(r, m) for m in MODES}
        all_r.append(r)
    pooled = {k: np.concatenate([r[k] for r in all_r]) for k in all_r[0]} if all_r else {}

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    out = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "config_hash": data.hash(),
        "seed": seed,
        "split": args.split,
        "climate": climate,
        "trajectories": names,
        "trajectories_missing": missing,
        "sampling": sampling_dict(s),
        "note": "Point summaries pooled over samples; trajectory-bootstrap intervals not yet "
        "computed ({{N_BOOT}} unresolved).",
        "pooled": {m: summarise(pooled, m) for m in MODES} if pooled else {},
        "per_trajectory": per_traj,
    }
    dest = REPO_ROOT / "results" / "midair" / args.split / "accuracy.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["pooled"], indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
