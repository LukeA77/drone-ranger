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
