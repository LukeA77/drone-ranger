"""End-to-end smoke test of scripts/midair_analyse.py on a tiny SYNTHETIC cache.

Catches crashes in any stage in seconds instead of at the end of a long real run.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_analyse  # noqa: E402
from test_midair_analysis import synthetic_cache  # noqa: E402


def test_analyse_runs_every_stage(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    names = []
    for g in range(3):  # three trajectories, so bins meet the minimum-groups rule
        r = synthetic_cache(n_frames=6, n_pix=60, seed=g)
        r.pop("group")
        n = len(r["u"])
        r["cls"] = np.where(np.arange(n) % 10 == 0, 8, 4)  # some water
        r["slope_deg"] = np.zeros(n)
        r["true_depression_deg"] = np.degrees(
            np.arctan2(
                r["true_point_w"][:, 2] - r["pose_p"][:, 2],
                np.hypot(*(r["true_point_w"][:, :2] - r["pose_p"][:, :2]).T),
            )
        )
        for mode in ("oracle_plane", "relative_altitude"):
            r[f"{mode}.valid"] = np.ones(n, bool)
            r[f"{mode}.range_est_m"] = r["true_range_m"] * 1.01
            r[f"{mode}.range_err_m"] = r["true_range_m"] * 0.01
            r[f"{mode}.rel_range_err"] = np.full(n, 0.01)
            r[f"{mode}.horiz_err_m"] = r["true_range_m"] * 0.01
        name = f"trajectory_{g:04d}"
        np.savez(cache / f"{name}.npz", **r)
        names.append(name)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"midair": {"splits": {"development": names}}}))
    out = tmp_path / "out"
    rc = midair_analyse.main(
        [
            "--split",
            "development",
            "--cache-dir",
            str(cache),
            "--splits-file",
            str(splits),
            "--out-dir",
            str(out),
            "--n-boot",
            "20",
            "--n-mc",
            "20",
            "--min-samples",
            "5",
            "--mc-per-bin",
            "5",
        ]
    )
    assert rc == 0
    for f in (
        "summary.json",
        "binned.json",
        "sweep.json",
        "propagation.json",
        "sensitivity.json",
        "slope.json",
        "flag.json",
    ):
        assert (out / f).exists(), f
    assert len(list((out / "figures").glob("*.png"))) >= 7


def test_overrides_refused_without_out_dir():
    assert midair_analyse.main(["--split", "development", "--n-boot", "5"]) == 1
