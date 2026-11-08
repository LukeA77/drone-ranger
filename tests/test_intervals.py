"""Calibrated intervals (pipeline/intervals.py)."""

import numpy as np
import pytest

from aerialrange.pipeline.intervals import IntervalCalibration, band

CFG = {
    "dep_edges": [0, 20, 40, 60, 91],
    "range_edges": [0, 50, 100, 200, np.inf],
    "range": {
        "pooled": {"0.68": 1.0, "0.90": 2.0, "0.95": 3.0},
        "cells": {"vehicle|d0|r2": {"0.68": 1.5, "0.90": 2.5, "0.95": 4.0}},
    },
    "speed": {"pooled": {"0.68": 1.2, "0.90": 2.2, "0.95": 2.9}, "cells": {}},
}


def test_band_includes_the_lower_edge_and_rejects_outside():
    e = (0.0, 20.0, 40.0)
    assert band(0.0, e) == 0 and band(20.0, e) == 1 and band(40.0, e) is None
    assert band(float("nan"), e) is None


def test_cell_multipliers_and_pooled_fallback():
    cal = IntervalCalibration.from_mapping(CFG)
    h = cal.half_widths("range", "vehicle", 10.0, 150.0, 2.0)
    assert h == pytest.approx({"0.68": 3.0, "0.90": 5.0, "0.95": 8.0})
    h = cal.half_widths("range", "person", 10.0, 150.0, 2.0)  # no cell: pooled
    assert h == pytest.approx({"0.68": 2.0, "0.90": 4.0, "0.95": 6.0})
    s = cal.half_widths("speed", "vehicle", 70.0, np.nan, 1.0)
    assert s["0.90"] == pytest.approx(2.2)
