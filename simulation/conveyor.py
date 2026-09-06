"""
The virtual conveyor.

Phase 1's stand-in for a belt motor plus a diverter. It owns nothing the
agents need to know about: items are placed on it, time advances, and items
that reach the end are handed to a bin.

Phase 2 replaces the *caller* of this class (SimulationController becomes
HardwareController) -- this file is simply not used any more. It is not a
component the agents import, which is why the swap is cheap.
"""

from __future__ import annotations

import logging

from core.config import Config
from core.messages import Destination
from simulation.bins import BinFarm
from simulation.waste_objects import ItemState, WasteItem

log = logging.getLogger(__name__)


class Conveyor:
    def __init__(self, cfg: Config, bins: BinFarm):
        self.bins = bins
        self.travel_seconds = float(cfg.get("conveyor.travel_ms", 2600)) / 1000.0
        self.running = bool(cfg.get("conveyor.running_on_start", True))
        self.stop_reason = ""
        self.items: list[WasteItem] = []
        self.delivered: list[WasteItem] = []

    # -- belt control ------------------------------------------------------

    def start(self) -> None:
        self.running = True
        self.stop_reason = ""

    def stop(self, reason: str = "operator request") -> None:
        self.running = False
        self.stop_reason = reason
        log.warning("Conveyor stopped: %s", reason)

    # -- items -------------------------------------------------------------

    def place(self, item: WasteItem) -> None:
        self.items.append(item)

    def hold(self, item_id: int, note: str) -> bool:
        for item in self.items:
            if item.item_id == item_id:
                item.state = ItemState.HELD
                item.note = note
                return True
        return False

    def release(self, item_id: int) -> bool:
        for item in self.items:
            if item.item_id == item_id and item.state is ItemState.HELD:
                item.state = ItemState.TRAVELLING
                item.note = ""
                return True
        return False

    # -- simulation tick ---------------------------------------------------

    def update(self, dt: float) -> list[WasteItem]:
        """Advance the belt by dt seconds. Returns items delivered this tick.

        A stopped belt freezes everything on it -- that is the whole point of
        the Safety Agent being able to stop it.
        """
        if not self.running:
            return []

        arrived: list[WasteItem] = []
        for item in self.items:
            item.advance(dt, self.travel_seconds)
            if item.arrived:
                arrived.append(item)

        for item in arrived:
            self._deliver(item)

        # Keep a short tail of delivered items so the dashboard can show
        # what just happened without holding the whole shift in memory.
        self.items = [i for i in self.items if i.state is not ItemState.DELIVERED]
        self.delivered = self.delivered[-50:]
        return arrived

    def _deliver(self, item: WasteItem) -> None:
        bin_ = self.bins.get(item.destination)
        if bin_ is None or not bin_.add(item.label):
            # Reaching the bin and finding it full is a late failure the
            # Safety Agent's pre-check should normally prevent. Hold rather
            # than drop, so nothing is silently lost.
            item.state = ItemState.HELD
            item.note = "bin full on arrival"
            log.warning("Item %s arrived at a full %s bin; holding",
                        item.item_id, item.destination.value)
            return
        item.state = ItemState.DELIVERED
        self.delivered.append(item)

    # -- reporting ---------------------------------------------------------

    @property
    def held_items(self) -> list[WasteItem]:
        return [i for i in self.items if i.state is ItemState.HELD]

    def snapshot(self) -> dict:
        return {
            "running": self.running,
            "stop_reason": self.stop_reason,
            "on_belt": len(self.items),
            "held": len(self.held_items),
        }
