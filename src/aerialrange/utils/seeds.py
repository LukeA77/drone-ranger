"""Seeding helpers for deterministic runs."""

from __future__ import annotations

import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Seed Python's `random` and return a NumPy generator for the given seed."""
    random.seed(seed)
    return np.random.default_rng(seed)
