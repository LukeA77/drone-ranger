"""Paired comparisons for H13 to H16 (scripts/multidrone16_paired.py, D-765)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multidrone16_paired as mp  # noqa: E402


def rows(times: list[float], crew: int = 0, horizon: float = 100.0) -> list[dict]:
    return [
        {
            "seed": i,
            "found": t is not None,
            "t": t,
            "horizon": horizon,
            "crew_before": crew,
            "bytes": 400,
            "steps": 100,
            "n": 4,
        }
        for i, t in enumerate(times)
    ]


def test_comparisons_keep_only_present_labels():
    names = [c[0] for c in mp.comparisons({"sweep_n4", "lawnmower_n4"})]
    assert names == ["H13_sweep_minus_lawnmower_n4"]


def test_paired_and_cost_differences(tmp_path, monkeypatch):
    data = {
        "sweep_n4": rows([10, 20, None]),  # unfound counts at the horizon
        "lawnmower_n4": rows([30, 40, 50], crew=5),
        "sweep_lose_any": rows([20, 30, None]),  # losing a drone costs the sweep 10 s
        "lawnmower_lose_any": rows([60, 70, 80], crew=5),  # and the lawnmower 30 s
    }
    monkeypatch.setattr(mp.mr, "OUT", tmp_path)
    (tmp_path / "phase16_test_seeds.json").write_text(json.dumps(data))
    assert mp.main(["test"]) == 0
    c = json.loads((tmp_path / "phase16_paired_test.json").read_text())["comparisons"]
    h13 = c["H13_sweep_minus_lawnmower_n4"]
    assert np.isclose(h13["time"]["mean"], ((10 - 30) + (20 - 40) + (100 - 50)) / 3)
    assert h13["crew"]["mean"] == -5 and h13["bytes"]["mean"] == 0
    cost = c["H15_lose_cost_sweep_minus_lawnmower"]
    # per seed: (20 - 60) - (10 - 30), (30 - 70) - (20 - 40), (100 - 80) - (100 - 50)
    assert np.isclose(cost["time"]["mean"], -70 / 3) and cost["time"]["excludes_0"]
