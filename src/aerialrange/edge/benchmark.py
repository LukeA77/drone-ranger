"""Per-stage latency benchmark.

A pipeline is a list of (name, callable) stages; each stage's output is the next stage's
input. Protocol sizes are `{{N_WARMUP}}`, `{{N_TIMED_FRAMES}}`, `{{N_REPEATS}}` from
configs/edge.yaml. Reports median, 95th and 99th percentiles per stage and end to end,
throughput, and the process's peak resident memory. Memory is read from the operating system
after timing, not traced during it: Python's allocation tracer slows every allocation and
would inflate the timings of Python-heavy stages such as tracking. Overlay drawing is not a
stage here (excluded from the budget).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import cycle, islice
from typing import Any

import numpy as np

Stage = tuple[str, Callable[[Any], Any]]


@dataclass(frozen=True)
class BenchmarkProtocol:
    n_warmup: int
    n_timed: int
    n_repeats: int

    def __post_init__(self) -> None:
        if self.n_timed < 1 or self.n_repeats < 1 or self.n_warmup < 0:
            raise ValueError("need n_timed >= 1, n_repeats >= 1, n_warmup >= 0")


def _stats(ms: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(ms)),
        "mean_ms": float(ms.mean()),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MiB (peak working set on Windows)."""
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 2**20
    import resource

    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20


def run_benchmark(
    stages: Sequence[Stage], inputs: Sequence[Any], protocol: BenchmarkProtocol
) -> dict[str, Any]:
    """Time each stage on `inputs` (cycled). Warm-up runs are not recorded."""
    names = [n for n, _ in stages]

    def one(x: Any) -> list[float]:
        t = []
        for _, fn in stages:
            t0 = time.perf_counter()
            x = fn(x)
            t.append((time.perf_counter() - t0) * 1000)
        return t

    for x in islice(cycle(inputs), protocol.n_warmup):
        one(x)
    rows, repeat_medians = [], []
    for _ in range(protocol.n_repeats):
        block = [one(x) for x in islice(cycle(inputs), protocol.n_timed)]
        repeat_medians.append(float(np.median(np.sum(block, axis=1))))
        rows += block
    t = np.array(rows)
    out = {n: _stats(t[:, i]) for i, n in enumerate(names)}
    out["end_to_end"] = _stats(t.sum(axis=1))
    return {
        "stages": out,
        "n_samples": len(rows),
        "throughput_fps": 1000.0 / out["end_to_end"]["mean_ms"],
        "end_to_end_median_by_repeat_ms": repeat_medians,
        "peak_rss_mb": peak_rss_mb(),
        "protocol": protocol.__dict__,
    }


def budget_verdict(stages: dict[str, dict[str, float]], budget_ms: float | None) -> dict[str, Any]:
    """Whether end-to-end latency meets `{{LATENCY_BUDGET_MS}}` at the median and the 99th
    percentile. Refuses to judge without a budget."""
    if budget_ms is None:
        raise ValueError("no latency budget: {{TARGET_FPS}} is unresolved")
    e = stages["end_to_end"]
    return {
        "budget_ms": budget_ms,
        "meets_at_median": e["median_ms"] <= budget_ms,
        "meets_at_p99": e["p99_ms"] <= budget_ms,
    }


def background_cpu(seconds: float = 3.0) -> dict[str, Any]:
    """CPU use of all other processes over `seconds`, measured just before timing, so each
    result records how quiet the machine actually was (in % of one core; the machine total is
    100 x logical cores). The five busiest are named."""
    import os

    import psutil

    me = os.getpid()
    procs = [p for p in psutil.process_iter(["name"]) if p.pid != me]
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(seconds)
    use = []
    for p in procs:
        try:
            use.append((p.info["name"] or "?", p.cpu_percent(None)))
        except psutil.Error:
            pass
    use = [u for u in use if u[0] not in ("System Idle Process", "Idle")]
    use.sort(key=lambda u: -u[1])
    return {
        "other_processes_cpu_percent_of_one_core": round(sum(u for _, u in use), 1),
        "busiest": [{"name": n, "cpu_percent": round(c, 1)} for n, c in use[:5]],
        "window_s": seconds,
    }
