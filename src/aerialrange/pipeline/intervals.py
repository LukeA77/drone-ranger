"""Calibrated range and speed intervals (D-737).

The filters and the ranging give a standard deviation per value; their errors are neither
Gaussian nor equally well described in every viewing condition. The calibration
(scripts/syndrone_interval_calibration.py) gives, per nominal level, a multiplier q so that the
interval is value +- q x sigma; q may differ by cell: class x line-of-sight depression band x
predicted-range band (speed: class x depression band). A cell without its own multiplier uses
the pooled one. Cell keys are "<group>|d<depression band>|r<range band>", bands numbered from 0
over the configured edges (lower edge included).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LEVEL_KEYS = ("0.68", "0.90", "0.95")


def band(x: float, edges: tuple[float, ...]) -> int | None:
    """Index of the band [edges[i], edges[i+1]) holding x; None outside or for NaN."""
    if not np.isfinite(x):
        return None
    i = int(np.searchsorted(edges, x, side="right")) - 1
    return i if 0 <= i < len(edges) - 1 else None


@dataclass(frozen=True)
class QuantityCalibration:
    pooled: Mapping[str, float]
    cells: Mapping[str, Mapping[str, float]]

    def multiplier(self, key: str | None, level: str) -> float:
        cell = self.cells.get(key) if key is not None else None
        return float(cell[level]) if cell and level in cell else float(self.pooled[level])


@dataclass(frozen=True)
class IntervalCalibration:
    dep_edges: tuple[float, ...]
    range_edges: tuple[float, ...]
    range: QuantityCalibration
    speed: QuantityCalibration

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> IntervalCalibration:
        def q(d: Mapping[str, Any]) -> QuantityCalibration:
            return QuantityCalibration(
                {str(k): float(v) for k, v in d["pooled"].items()},
                {
                    c: {str(k): float(v) for k, v in lv.items()}
                    for c, lv in d.get("cells", {}).items()
                },
            )

        return cls(
            tuple(float(x) for x in m["dep_edges"]),
            tuple(float(x) for x in m["range_edges"]),
            q(m["range"]),
            q(m["speed"]),
        )

    def key(self, group: str, dep_deg: float, range_m: float | None) -> str | None:
        d = band(dep_deg, self.dep_edges)
        if d is None:
            return None
        r = 0 if range_m is None else band(range_m, self.range_edges)
        return None if r is None else f"{group}|d{d}|r{r}"

    def half_widths(
        self, quantity: str, group: str, dep_deg: float, range_m: float, sigma: float
    ) -> dict[str, float]:
        """Half-widths at each level for one value with standard deviation `sigma`."""
        qc = self.range if quantity == "range" else self.speed
        k = self.key(group, dep_deg, range_m if quantity == "range" else None)
        return {lv: qc.multiplier(k, lv) * sigma for lv in LEVEL_KEYS}


def config_from_calibration(path: str | Path) -> dict[str, Any]:
    """The `intervals` configuration from a calibration result
    (results/syndrone/calibration/interval_calibration_<run>.json). The multipliers hold for
    the conditions they were fitted in (SynDrone: exact camera poses), so they are passed
    explicitly for those runs and are not part of the general run configuration."""
    res = json.loads(Path(path).read_text())
    q = res["quantities"]
    return {
        "dep_edges": res["dep_edges"],
        "range_edges": res["range_edges"],
        "range": q["range"]["fitted"],
        "speed": q["speed"]["fitted"],
    }
