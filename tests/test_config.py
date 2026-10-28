import pytest

from aerialrange.utils.config import (
    CONFIG_DIR,
    ConfigFormatError,
    MissingParameterError,
    latency_budget_ms,
    load_config,
    parse_config,
)

CONFIG_NAMES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_shipped_config_parses(name):
    cfg = load_config(name)
    assert cfg.params


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_resolved_values_record_a_source(name):
    """Any value that is set must say where it came from."""
    cfg = load_config(name)
    for key, p in cfg.params.items():
        if p.resolved:
            assert p.source, f"{name}.{key} has a value but no source"


def test_require_raises_on_null_with_placeholder_name():
    cfg = parse_config("x", {"k": {"value": None, "placeholder": "FOO", "source": None}})
    with pytest.raises(MissingParameterError, match=r"\{\{FOO\}\}"):
        cfg.require("k")


def test_require_returns_resolved_value():
    cfg = parse_config("x", {"k": {"value": 3, "placeholder": "FOO", "source": "test"}})
    assert cfg.require("k") == 3
    assert cfg.unresolved() == []


def test_unknown_key_raises():
    cfg = parse_config("x", {})
    with pytest.raises(KeyError):
        cfg.get("nope")


def test_malformed_entry_rejected():
    with pytest.raises(ConfigFormatError):
        parse_config("x", {"k": 5})


def test_latency_budget_derived_from_fps():
    cfg = parse_config("edge", {"target_fps": {"value": 20, "placeholder": "T", "source": "t"}})
    assert latency_budget_ms(cfg) == pytest.approx(50.0)


def test_latency_budget_refuses_unset_fps():
    entry = {"value": None, "placeholder": "TARGET_FPS", "source": None}
    unset = parse_config("edge", {"target_fps": entry})
    with pytest.raises(MissingParameterError):
        latency_budget_ms(unset)


def test_latency_budget_is_derived_from_target_fps():
    assert latency_budget_ms(load_config("edge")) == pytest.approx(1000 / 30)


def test_config_hash_changes_with_values():
    a = parse_config("x", {"k": {"value": 1, "placeholder": None, "source": "s"}})
    b = parse_config("x", {"k": {"value": 2, "placeholder": None, "source": "s"}})
    assert a.hash() != b.hash()
