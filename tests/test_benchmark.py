import time

import numpy as np
import pytest

from aerialrange.edge.benchmark import BenchmarkProtocol, budget_verdict, run_benchmark


def _sleepy(ms):
    def f(x):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ms / 1000:
            pass
        return x

    return f


def test_stage_timings_and_percentiles():
    stages = [("a", _sleepy(2)), ("b", _sleepy(1))]
    proto = BenchmarkProtocol(n_warmup=2, n_timed=30, n_repeats=2)
    r = run_benchmark(stages, inputs=[0] * 5, protocol=proto)
    assert set(r["stages"]) == {"a", "b", "end_to_end"}
    a = r["stages"]["a"]
    assert a["median_ms"] == pytest.approx(2.0, rel=0.3)
    assert a["p95_ms"] >= a["median_ms"] and a["p99_ms"] >= a["p95_ms"]
    e = r["stages"]["end_to_end"]
    assert e["median_ms"] == pytest.approx(3.0, rel=0.3)
    assert r["throughput_fps"] == pytest.approx(1000 / e["mean_ms"], rel=1e-6)
    assert r["n_samples"] == 60
    assert (
        r["peak_rss_mb"] > 0
        and len(r["end_to_end_median_by_repeat_ms"]) == r["protocol"]["n_repeats"]
    )


def test_stage_output_feeds_next_stage():
    seen = []
    stages = [("double", lambda x: 2 * x), ("record", lambda x: seen.append(x) or x)]
    run_benchmark(stages, inputs=[1, 2], protocol=BenchmarkProtocol(0, 2, 1))
    assert seen == [2, 4]


def test_budget_verdict_uses_median_and_p99():
    stages = {"end_to_end": {"median_ms": 30.0, "p99_ms": 45.0}}
    v = budget_verdict(stages, budget_ms=40.0)
    assert v == {"budget_ms": 40.0, "meets_at_median": True, "meets_at_p99": False}
    with pytest.raises(ValueError):
        budget_verdict(stages, budget_ms=None)


def test_protocol_validation():
    with pytest.raises(ValueError):
        BenchmarkProtocol(n_warmup=0, n_timed=0, n_repeats=1)
    assert np.isfinite(BenchmarkProtocol(1, 1, 1).n_timed)
