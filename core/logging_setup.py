"""Logging: a human-readable log file plus a machine-readable event stream."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import Config
from core.event_bus import EventBus
from core.messages import Event, Severity

_SEVERITY_TO_LEVEL = {
    Severity.INFO: logging.INFO,
    Severity.WARNING: logging.WARNING,
    Severity.ALERT: logging.ERROR,
    Severity.CRITICAL: logging.CRITICAL,
}


def setup_logging(cfg: Config) -> None:
    level = getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO)
    log_file = cfg.path("logging.file", "logs/system.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    to_file = logging.FileHandler(log_file, encoding="utf-8")
    to_file.setFormatter(fmt)
    root.addHandler(to_file)

    # Ultralytics logs every inference at INFO and would drown the agent
    # narrative, which is the part worth watching in a demo.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


class EventRecorder:
    """Writes every bus event to JSONL and mirrors a summary into the log.

    Stage 3 swaps this sink for SQLite. The subscription does not change,
    which is the point of hanging persistence off the bus rather than
    wiring it into the agents.
    """

    def __init__(self, cfg: Config, bus: EventBus):
        self.path: Path = cfg.path("logging.json_events", "logs/events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger("events")
        self._fh = open(self.path, "a", encoding="utf-8")
        bus.subscribe("*", self._on_event)

    def _on_event(self, event: Event) -> None:
        record = event.to_dict()
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self.log.log(
            _SEVERITY_TO_LEVEL.get(event.severity, logging.INFO),
            "%-20s %s", event.event, _summarise(record),
        )

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


_INTERESTING = ("label", "material", "category", "action", "destination",
                "confidence", "status", "rule", "bin", "reason")


def _summarise(record: dict) -> str:
    payload = record.get("payload", {})
    bits = []
    for key in _INTERESTING:
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, float):
            value = f"{value:.2f}"
        bits.append(f"{key}={value}")
    return "  ".join(bits)
