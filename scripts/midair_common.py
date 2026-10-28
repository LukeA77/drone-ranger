"""Shared helpers for the Mid-Air scripts: loading a split's cache and the noise sweep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerialrange.utils.config import REPO_ROOT, load_config

CACHE = REPO_ROOT / "cache" / "midair"
DEG = np.pi / 180


SPLITS = REPO_ROOT / "results" / "splits.json"


def split_names(split: str, splits_file: Path = SPLITS) -> list[str]:
    return json.loads(splits_file.read_text())["midair"]["splits"][split]


def load_split_cache(
    split: str, cache_dir: Path = CACHE, splits_file: Path = SPLITS
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Pool cached samples of a split's evaluated trajectories; `group` = trajectory number."""
    parts, names = [], []
    for name in split_names(split, splits_file):
        f = cache_dir / f"{name}.npz"
        if f.exists():
            d = dict(np.load(f))
            d["group"] = np.full(len(d["u"]), int(name.split("_")[-1]))
            parts.append(d)
            names.append(name)
    if not parts:
        raise FileNotFoundError(f"no cached trajectories for split {split!r}; run midair_eval")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, names


def noise_levels() -> dict[str, dict[str, float]]:
    """The Mid-Air sensor-noise sweep from configs/uncertainty.yaml, in internal units
    (metres, radians, pixels). The config stores angles in degrees and focal-length noise
    as a fraction of the focal length (fx = fy = 512 px for 1024 px Mid-Air images)."""
    cfg = load_config("uncertainty").require("midair_noise_sweep")
    out = {}
    for name, lv in cfg.items():
        f = 512.0 * lv["focal_frac"]
        out[name] = {
            "u": lv["pixel_px"],
            "v": lv["pixel_px"],
            "h": lv["height_m"],
            "roll": lv["attitude_deg"] * DEG,
            "pitch": lv["attitude_deg"] * DEG,
            "yaw": lv["attitude_deg"] * DEG,
            "tilt": lv["tilt_deg"] * DEG,
            "fx": f,
            "fy": f,
        }
    return out
