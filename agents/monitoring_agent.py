"""
AGENT 8 -- MONITORING AGENT

Watches the system rather than the waste.

It subscribes to the bus; nothing calls it. The pipeline publishes
WASTE_RECORDED when an item is finished with and never learns who listened.
That is what makes Stage 4's Analytics and Planning agents additions rather
than edits: they subscribe to the same events and the sorting path does not
move.

It also owns the two jobs that are about elapsed time rather than about any
one item -- sampling bin levels, and noticing when a bin crosses its warning
threshold. Both were briefly done at commit time in Stage 1, which was wrong:
an item does not land in its bin at the moment it is decided, it lands
several seconds later when it reaches the end of the belt. A threshold
crossing was therefore only noticed when the *next* item happened to be
committed, which on a quiet line could be minutes.
"""

from __future__ import annotations

import time
from collections import Counter

from core.config import Config
from core.event_bus import WILDCARD, EventBus
from core.messages import Action, ActionStatus, Event, Severity, Topic, WasteRecord

from agents.base import Agent


class Stats:
    """Live counters. Cheap, in memory, and independent of the database.

    They are kept even when persistence has failed, because the operator
    looking at the window still needs to know what the shift has done.
    """

    def __init__(self):
        self.by_category: Counter[str] = Counter()
        self.by_rule: Counter[str] = Counter()
        self.by_destination: Counter[str] = Counter()
        self.processed = 0
        self.sorted_ok = 0
        self.manual_checks = 0
        self.failures = 0
        self._confidence_total = 0.0

    def record(self, record: WasteRecord) -> None:
        self.processed += 1
        self.by_category[record.category.value] += 1
        self.by_rule[record.safety_rule] += 1
        self.by_destination[record.destination.value] += 1
        self._confidence_total += record.detection_confidence
        if record.status is ActionStatus.OK:
            self.sorted_ok += 1
        else:
            self.failures += 1
        if record.action is Action.MANUAL_CHECK:
            self.manual_checks += 1

    @property
    def average_confidence(self) -> float:
        return self._confidence_total / self.processed if self.processed else 0.0

    @property
    def manual_rate(self) -> float:
        return self.manual_checks / self.processed if self.processed else 0.0


class MonitoringAgent(Agent):
    name = "MonitoringAgent"

    def __init__(self, cfg: Config, bus: EventBus, world, db=None):
        super().__init__(bus)
        self.world = world
        self.db = db
        self.stats = Stats()
        self.records: list[WasteRecord] = []

        self.bin_sample_seconds = float(cfg.get("database.bin_sample_seconds", 10))
        self.heartbeat_seconds = float(cfg.get("monitoring.heartbeat_seconds", 10))

        self._warned_bins: set[str] = set()
        self._last_bin_sample = 0.0
        self._last_heartbeat = 0.0

        bus.subscribe(Topic.WASTE_RECORDED, self._on_record)
        bus.subscribe(WILDCARD, self._on_any)

    # -- subscriptions -----------------------------------------------------

    def _on_record(self, event: Event) -> None:
        record = event.payload.get("record")
        if not isinstance(record, WasteRecord):
            return
        self.stats.record(record)
        self.records.append(record)
        if self.db is not None:
            self.db.insert_detection(record)

    def _on_any(self, event: Event) -> None:
        if self.db is not None:
            self.db.insert_event(event)

    # -- timed work --------------------------------------------------------

    def tick(self, now: float | None = None, at: str | None = None) -> None:
        """Called once per frame from the main loop. Cheap on most frames.

        `at` is only used by scripted runs, which advance a simulated clock
        far faster than wall time; the live system passes neither argument.
        """
        now = time.monotonic() if now is None else now
        self._check_bins()

        if self.db is not None and now - self._last_bin_sample >= self.bin_sample_seconds:
            self._last_bin_sample = now
            self.db.snapshot_bins(self.world.bins, at)

        if now - self._last_heartbeat >= self.heartbeat_seconds:
            self._last_heartbeat = now
            self.emit(Topic.SYSTEM_UPDATE, {
                "processed": self.stats.processed,
                "sorted_ok": self.stats.sorted_ok,
                "manual_checks": self.stats.manual_checks,
                "failures": self.stats.failures,
                "average_confidence": round(self.stats.average_confidence, 3),
                "conveyor_running": self.world.conveyor.running,
            })

    def _check_bins(self) -> None:
        """Alert on a threshold crossing, once, not on every frame after it.

        The key includes the status, so a bin that goes from NEARING CAPACITY
        to FULL raises a second, louder alert rather than staying quiet
        because it had already warned once.
        """
        for bin_ in self.world.bins.warnings():
            key = f"{bin_.destination.value}:{bin_.status}"
            if key in self._warned_bins:
                continue
            self._warned_bins.add(key)
            self.emit(Topic.BIN_ALERT, {
                "bin": bin_.destination.value,
                "percent": bin_.percent,
                "capacity": bin_.capacity,
                "level": bin_.current_level,
                "reason": f"{bin_.destination.value} bin {bin_.status.lower()} "
                          f"({bin_.percent}%)",
            }, Severity.ALERT if bin_.is_full else Severity.WARNING)

        # Emptying a bin re-arms its alerts.
        still_warning = {f"{b.destination.value}:{b.status}"
                         for b in self.world.bins.warnings()}
        self._warned_bins &= still_warning

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "processed": self.stats.processed,
            "sorted_ok": self.stats.sorted_ok,
            "manual_checks": self.stats.manual_checks,
            "failures": self.stats.failures,
            "manual_rate": round(self.stats.manual_rate, 3),
            "average_confidence": round(self.stats.average_confidence, 3),
            "by_category": dict(self.stats.by_category),
            "by_rule": dict(self.stats.by_rule),
            "by_destination": dict(self.stats.by_destination),
            "persistence": bool(self.db is not None and self.db.available),
            "world": self.world.snapshot(),
        }
