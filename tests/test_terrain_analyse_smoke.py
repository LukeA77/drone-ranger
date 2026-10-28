"""End-to-end smoke test of scripts/terrain_analyse.py on SYNTHETIC caches (all modes)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import terrain_analyse as ta  # noqa: E402
from test_midair_analysis import synthetic_cache  # noqa: E402


def _write_split(cache, names, seed0):
    rng = np.random.default_rng(seed0)
    for i, name in enumerate(names):
        r = synthetic_cache(n_frames=8, n_pix=80, seed=seed0 + i)
        r.pop("group")
        n = len(r["u"])
        t = r["true_range_m"]
        r["cls"] = np.full(n, 4)
        r["slope_deg"] = rng.uniform(0, 30, n)
        r["true_depression_deg"] = rng.uniform(1, 60, n)
        for mode in ("oracle_plane", "relative_altitude"):
            r[f"{mode}.valid"] = np.ones(n, bool)
            r[f"{mode}.range_est_m"] = t * (1 + rng.normal(0, 0.1, n))
        np.savez(cache / f"{name}.npz", **r)
        sigma = 0.05 * t
        est = t + 2.0 * sigma * rng.normal(size=n)  # sigma under-states the error by 2x
        terr = {
            "terrain.valid": np.ones(n, bool),
            "terrain.range_m": est,
            "terrain.range_sigma_m": sigma,
            "terrain.roughness_m": rng.uniform(0, 1, n),
            "terrain.support_radius_m": rng.uniform(1, 10, n),
            "terrain.n_points": rng.integers(100, 1000, n),
            "terrain.point_w": r["true_point_w"],
        }
        np.savez(cache / f"{name}.terrain.npz", **terr)


def test_compare_fit_and_apply(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    splits = {
        "development": ["trajectory_0001", "trajectory_0002", "trajectory_0003"],
        "calibration": ["trajectory_0004", "trajectory_0005", "trajectory_0006"],
        "test": ["trajectory_0007", "trajectory_0008", "trajectory_0009"],
    }
    for k, names in enumerate(splits.values()):
        _write_split(cache, names, seed0=10 * k)
    monkeypatch.setattr(ta, "CACHE", cache)
    monkeypatch.setattr(ta, "split_names", lambda s: splits[s])
    monkeypatch.setattr(ta, "FIT", tmp_path / "fit.json")
    monkeypatch.setattr(ta, "REPO_ROOT", tmp_path)
    assert ta.main(["--split", "development"]) == 0
    out = tmp_path / "results" / "midair" / "development" / "terrain"
    summ = json.loads((out / "summary.json").read_text())["data"]
    assert set(summ) == {"flat_true_height", "flat_altimeter", "terrain_motion"}
    assert ta.main(["--fit-calibration"]) == 0
    fit = json.loads((tmp_path / "fit.json").read_text())
    assert fit["conformal"]["global"]["0.68"] > 1.5  # recovers the 2x under-statement
    assert ta.main(["--split", "test"]) == 1  # refused without --final
    assert ta.main(["--split", "test", "--final"]) == 0
    cal = json.loads(
        (tmp_path / "results" / "midair" / "test" / "terrain" / "calibration.json").read_text()
    )["data"]
    assert abs(cal["conformal"]["0.68"] - 0.68) < 0.08
    assert cal["raw"]["coverage"]["0.68"] < 0.5
