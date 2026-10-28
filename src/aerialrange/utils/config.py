"""Load `configs/*.yaml` into typed objects and refuse to use unresolved placeholders.

Each config entry has the form::

    name:
      value: <resolved value or null>
      placeholder: <PLACEHOLDER_NAME, or null>
      source: <where the value came from, or null>

A `null` value means the parameter is unresolved. Code that needs it calls
`Config.require`, which raises `MissingParameterError` naming the placeholder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class MissingParameterError(RuntimeError):
    """Raised when a command needs a parameter that is still a `{{PLACEHOLDER}}`."""


class ConfigFormatError(ValueError):
    """Raised when a config file does not follow the value/placeholder/source format."""


@dataclass(frozen=True)
class Param:
    """One configuration entry with its provenance."""

    value: Any
    placeholder: str | None
    source: str | None

    @property
    def resolved(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class Config:
    """A parsed config file: a name and its parameters."""

    name: str
    params: dict[str, Param]

    def get(self, key: str) -> Any:
        """Return the value of `key` (may be None if unresolved)."""
        return self._param(key).value

    def require(self, key: str) -> Any:
        """Return the value of `key`, raising `MissingParameterError` if it is null."""
        p = self._param(key)
        if p.value is None:
            tag = f"{{{{{p.placeholder}}}}}" if p.placeholder else "(no placeholder name)"
            raise MissingParameterError(
                f"{self.name}.{key} is unresolved {tag}. Measure it from data or cite a "
                f"primary source, and record the source in configs/{self.name}.yaml."
            )
        return p.value

    def unresolved(self) -> list[str]:
        """Names of parameters whose value is still null."""
        return [k for k, p in self.params.items() if not p.resolved]

    def _param(self, key: str) -> Param:
        if key not in self.params:
            raise KeyError(f"{self.name}.{key} is not defined in configs/{self.name}.yaml")
        return self.params[key]

    def hash(self) -> str:
        """Short SHA-256 of the parameter values, for results provenance."""
        payload = json.dumps({k: p.value for k, p in sorted(self.params.items())}, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def parse_config(name: str, raw: dict[str, Any]) -> Config:
    """Validate a raw YAML mapping and build a `Config`."""
    if not isinstance(raw, dict):
        raise ConfigFormatError(f"{name}: top level must be a mapping")
    params: dict[str, Param] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or set(entry) != {"value", "placeholder", "source"}:
            raise ConfigFormatError(
                f"{name}.{key}: expected keys value, placeholder, source; got {entry!r}"
            )
        params[key] = Param(entry["value"], entry["placeholder"], entry["source"])
    return Config(name, params)


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> Config:
    """Load `configs/<name>.yaml`."""
    path = config_dir / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return parse_config(name, raw)


def latency_budget_ms(edge: Config) -> float:
    """Per-frame budget in milliseconds, derived as 1000 / target_fps."""
    fps = edge.require("target_fps")
    return 1000.0 / float(fps)
