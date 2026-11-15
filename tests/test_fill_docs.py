import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fill_docs as fd  # noqa: E402


@pytest.fixture
def results(tmp_path):
    (tmp_path / "midair").mkdir()
    (tmp_path / "midair" / "summary.json").write_text(
        json.dumps(
            {
                "data": {
                    "oracle_plane": {
                        "land": {"abs_rel_err_median": 0.10534, "n": 223000, "ci": [0.08, 0.12]}
                    }
                }
            }
        )
    )
    return tmp_path


def test_replaces_token_with_formatted_value(results):
    text = (
        "Median error {{RESULT:midair/summary.json#data.oracle_plane.land.abs_rel_err_median|.1%}}."
    )
    assert fd.fill(text, results) == "Median error 10.5%."


def test_integer_and_list_index_and_default_format(results):
    t = (
        "n = {{RESULT:midair/summary.json#data.oracle_plane.land.n|,}}; "
        "lo {{RESULT:midair/summary.json#data.oracle_plane.land.ci.0}}"
    )
    assert fd.fill(t, results) == "n = 223,000; lo 0.08"


def test_missing_value_raises(results):
    with pytest.raises(fd.FillError, match="nope"):
        fd.fill("{{RESULT:midair/summary.json#data.nope}}", results)


def test_unresolved_placeholders_are_reported():
    assert fd.unresolved("a {{TARGET_FPS}} b {{RESULT:x#y}} c") == [
        "{{TARGET_FPS}}",
        "{{RESULT:x#y}}",
    ]


def test_keys_containing_dots(tmp_path):
    (tmp_path / "e.json").write_text(
        json.dumps({"vehicle": {"pooled": {"range_coverage_0.90": 0.982}, "ci": {"x": [1, 2]}}})
    )
    t = "{{RESULT:e.json#vehicle.pooled.range_coverage_0.90|.0%}} {{RESULT:e.json#vehicle.ci.x.1}}"
    assert fd.fill(t, tmp_path) == "98% 2"
    with pytest.raises(fd.FillError, match="range_coverage_0"):
        fd.fill("{{RESULT:e.json#vehicle.pooled.range_coverage_0.95}}", tmp_path)


def test_neg_prefix_negates_before_formatting(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"diff": -169.25}))
    assert fd.fill("{{RESULT:x.json#diff|neg:,.0f}} s saved", tmp_path) == "169 s saved"


def test_min_prefix_turns_seconds_into_minutes(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"t": 1697.0}))
    assert fd.fill("after {{RESULT:x.json#t|min:.0f}} minutes", tmp_path) == "after 28 minutes"
