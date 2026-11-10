"""Navigation grades added to SynDrone poses (syndrone_refilter.py --nav-grade, Phase 10)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import syndrone_refilter as sr  # noqa: E402
from terrain_pose_sweep import CONDITIONS  # noqa: E402

N = 3000  # frames, as many as a SynDrone sequence


def flight(n: int = N) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": range(1, n + 1),
            "t_s": np.arange(n) / 25.0,
            "n_m": np.zeros(n),
            "e_m": np.zeros(n),
            "h_m": np.full(n, 50.0),
            "yaw_rad": np.zeros(n),
            "tilt_deg": np.full(n, 45.0),
        }
    ).set_index("frame")


def test_grades_are_version_1s_values():
    assert sr.GRADES == ("nav_good", "nav_typical", "nav_poor")
    assert CONDITIONS["nav_good"] == (0.02, 0.02, 1.0, 0.25)
    assert CONDITIONS["nav_poor"] == (0.2, 0.2, 5.0, 1.0)


def test_town_number():
    assert sr.town_number("Town01") == 1
    assert sr.town_number("Town10HD") == 10


def test_perturbation_is_deterministic_and_leaves_input_alone():
    m = flight(50)
    a = sr.perturb_flight(m, "nav_typical", "Town02", 50, 0)
    b = sr.perturb_flight(m, "nav_typical", "Town02", 50, 0)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(m, flight(50))
    c = sr.perturb_flight(m, "nav_typical", "Town02", 80, 0)
    assert not np.allclose(a["n_m"], c["n_m"])  # another sequence, other draws


def test_errors_have_the_grades_sizes():
    pj, aj, pb, ab = CONDITIONS["nav_poor"]
    m = flight()
    p = sr.perturb_flight(m, "nav_poor", "Town03", 20, 0)
    dn = p["n_m"] - m["n_m"]
    dt = p["tilt_deg"] - m["tilt_deg"]
    dy = np.degrees(p["yaw_rad"] - m["yaw_rad"])
    # within a sequence the bias is constant and the jitter has the grade's spread
    for d, sd in ((dn, pj), (dt, aj), (dy, aj)):
        assert abs(np.std(d) / sd - 1) < 0.1
    # across many sequences the bias has the grade's spread
    biases = [
        sr.perturb_flight(flight(2), "nav_poor", "Town03", h, 0)["n_m"].mean() for h in range(400)
    ]
    assert abs(np.std(biases) / math.hypot(pb, pj / math.sqrt(2)) - 1) < 0.15
    assert ab == 1.0


def test_columns_and_times_unchanged():
    m = flight(10)
    p = sr.perturb_flight(m, "nav_good", "Town06", 50, 0)
    assert list(p.columns) == list(m.columns)
    pd.testing.assert_series_equal(p["t_s"], m["t_s"])
