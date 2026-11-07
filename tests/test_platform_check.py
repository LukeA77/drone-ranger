"""The cross-platform output check passes identical runs and fails a real difference."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import platform_check  # noqa: E402


def _run(tag):
    rows = [[f, 1, 3, 100.0 + 5 * f, 50.0, 20.0, 10.0, 80.0 + f, True] for f in range(1, 11)] + [
        [f, 2, 0, 300.0, 200.0, 8.0, 16.0, 120.0, True] for f in range(1, 11)
    ]
    dets = [
        {
            "frame": r[0],
            "left": r[3],
            "top": r[4],
            "width": r[5],
            "height": r[6],
            "score": 0.8,
            "cls": r[2],
        }
        for r in rows
    ]
    return {
        "tag": tag,
        "machine": {},
        "detector": {"sha256": "x"},
        "frames": "f",
        "demo": {"rows": rows},
        "real": {"detections": dets, "rows": copy.deepcopy(rows)},
    }


def _write(tmp_path, data):
    p = tmp_path / f"{data['tag']}.json"
    p.write_text(json.dumps(data))
    return p


def test_identical_runs_pass_and_ids_may_be_relabelled(tmp_path):
    a, b = _run("a"), _run("b")
    for r in b["real"]["rows"]:
        r[1] = {1: 7, 2: 9}[r[1]]  # different id labels, same tracks
    assert platform_check.compare(_write(tmp_path, a), _write(tmp_path, b), tmp_path) == 0


def test_a_shifted_box_fails(tmp_path):
    a, b = _run("a"), _run("b")
    b["real"]["detections"][3]["left"] += 1.0  # beyond the 0.5 px tolerance
    assert platform_check.compare(_write(tmp_path, a), _write(tmp_path, b), tmp_path) == 1
    out = json.loads((tmp_path / "platform_compare_a_b.json").read_text())
    assert out["checks"]["detection_boxes"] is False and out["passed"] is False


def test_an_identity_switch_fails(tmp_path):
    a, b = _run("a"), _run("b")
    for r in b["demo"]["rows"]:
        if r[0] > 5:
            r[1] = 3 - r[1]  # the two tracks swap ids halfway
    assert platform_check.compare(_write(tmp_path, a), _write(tmp_path, b), tmp_path) == 1
