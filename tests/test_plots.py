import csv

import pytest

from aerialrange.viz.plots import bar_chart, line_chart


def test_line_chart_writes_png_and_csv(tmp_path):
    p = tmp_path / "fig.png"
    line_chart(
        p,
        {
            "a": {"x": [1, 2], "y": [3, 4], "lo": [2, 3], "hi": [4, 5]},
            "b": {"x": [1, 2], "y": [1, float("nan")]},
        },
        "t",
        "x",
        "y",
        logx=True,
    )
    assert p.stat().st_size > 1000
    rows = list(csv.DictReader(p.with_suffix(".csv").open()))
    assert len(rows) == 4 and rows[0]["series"] == "a"


def test_line_chart_caps_series():
    with pytest.raises(ValueError):
        line_chart(None, {k: {"x": [1], "y": [1]} for k in "abcd"}, "t", "x", "y")


def test_bar_chart(tmp_path):
    p = tmp_path / "bar.png"
    bar_chart(p, ["h", "pitch"], [0.5, 2.0], "t", "x", note="n")
    assert p.exists() and p.with_suffix(".csv").exists()


def test_pareto_frontier_keeps_only_undominated_points():
    from aerialrange.viz.plots import pareto_frontier

    pts = [(10, 0.5), (20, 0.6), (15, 0.4), (30, 0.6), (5, 0.3)]
    assert pareto_frontier(pts) == [4, 0, 1]


def test_pareto_chart_writes_png_and_csv(tmp_path):
    from aerialrange.viz.plots import pareto_chart

    p = tmp_path / "pareto.png"
    pareto_chart(p, ["a", "b", "c"], [10, 20, 15], [0.5, 0.6, 0.4], {"30 fps": 33.3}, "t", "x", "y")
    assert p.exists() and p.with_suffix(".csv").read_text().count("\n") == 4
