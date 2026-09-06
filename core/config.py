"""Configuration loading. One YAML file, dot access, paths resolved once."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class Config:
    """Read-only view over the YAML tree.

    Access is by dotted path so callers read like prose:

        cfg.get("confidence.auto_sort")
        cfg.get("action.controller")
    """

    def __init__(self, data: dict[str, Any], source: Path):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        if not path.exists():
            raise FileNotFoundError(
                f"Config not found at {path}. Expected the project's "
                f"config/config.yaml -- restore it or pass --config."
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"Required config key '{dotted}' missing from {self.source}")
        return value

    def path(self, dotted: str, default: str | None = None) -> Path:
        """A config value interpreted as a path relative to the project root."""
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"Config key '{dotted}' missing and no default given")
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    def __repr__(self) -> str:
        return f"<Config {self.source.name} keys={list(self._data)}>"


def camera_source(cfg: Config) -> int | str:
    """Webcam index or video-file path, resolved and validated.

    A bare integer means a device index; anything else is a file we check
    exists now rather than failing three layers down inside OpenCV.
    """
    raw = cfg.get("camera.source", 0)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"camera.source points at {raw!r} but no such file exists "
            f"(looked at {p}). Use 0 for the default webcam."
        )
    return str(p)
