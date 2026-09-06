"""Phase 1 actuation: drives the virtual conveyor and bins."""

from __future__ import annotations

import logging
import random
import time

from core.config import Config
from core.messages import ActionResult, ActionStatus, Destination
from hardware.controller import ActionController

log = logging.getLogger(__name__)


class SimulationController(ActionController):
    """Executes decisions against the simulation.

    `simulated_failure_rate` in config is not decoration. A simulated belt
    never jams, so without injected failures the retry-then-hold path in the
    Action Agent would be dead code that first runs the day real hardware is
    attached. Set it to 0.2 and watch the recovery logic work before an
    ESP32 exists.
    """

    name = "simulation"

    def __init__(self, cfg: Config, simulation):
        self.sim = simulation                      # SimulationController's view of the world
        self.failure_rate = float(cfg.get("action.simulated_failure_rate", 0.0))
        self._rng = random.Random(20260905)         # deterministic demos
        self.commands: list[str] = []

    # -- helpers -----------------------------------------------------------

    def _injected_failure(self) -> ActionStatus | None:
        if self.failure_rate <= 0 or self._rng.random() >= self.failure_rate:
            return None
        # Split injected failures between the two modes real hardware shows.
        return ActionStatus.TIMEOUT if self._rng.random() < 0.3 else ActionStatus.FAILED

    @staticmethod
    def _elapsed(started: float) -> float:
        return (time.perf_counter() - started) * 1000.0

    # -- ActionController --------------------------------------------------

    def sort_to(self, destination: Destination, item_id: int) -> ActionResult:
        started = time.perf_counter()
        self.commands.append(f"sort_to({destination.value}, item={item_id})")

        if not self.sim.conveyor.running:
            return ActionResult(
                ActionStatus.REFUSED, destination, self._elapsed(started),
                detail="conveyor is stopped",
            )

        if self.sim.bins.is_full(destination):
            return ActionResult(
                ActionStatus.REFUSED, destination, self._elapsed(started),
                detail=f"{destination.value} bin is full",
            )

        failure = self._injected_failure()
        if failure is not None:
            return ActionResult(
                failure, destination, self._elapsed(started),
                detail="injected actuator failure (simulated_failure_rate)",
            )

        # In the simulation, "sorting" means the item is already travelling
        # toward this destination; delivery happens when it reaches the bin.
        return ActionResult(ActionStatus.OK, destination, self._elapsed(started),
                            detail="diverter set")

    def start_conveyor(self) -> ActionResult:
        started = time.perf_counter()
        self.commands.append("start_conveyor()")
        self.sim.conveyor.start()
        return ActionResult(ActionStatus.OK, Destination.NONE, self._elapsed(started))

    def stop_conveyor(self, reason: str = "") -> ActionResult:
        started = time.perf_counter()
        self.commands.append(f"stop_conveyor({reason})")
        self.sim.conveyor.stop(reason or "unspecified")
        return ActionResult(ActionStatus.OK, Destination.NONE, self._elapsed(started),
                            detail=reason)

    def hold_item(self, item_id: int, reason: str = "") -> ActionResult:
        started = time.perf_counter()
        self.commands.append(f"hold_item({item_id}, {reason})")
        held = self.sim.conveyor.hold(item_id, reason)
        status = ActionStatus.OK if held else ActionStatus.FAILED
        detail = reason if held else f"no item {item_id} on the belt"
        return ActionResult(status, Destination.NONE, self._elapsed(started), detail=detail)

    def check_status(self) -> dict:
        return {
            "available": True,
            "controller": self.name,
            "conveyor_running": self.sim.conveyor.running,
            "stop_reason": self.sim.conveyor.stop_reason,
            "commands_issued": len(self.commands),
        }
