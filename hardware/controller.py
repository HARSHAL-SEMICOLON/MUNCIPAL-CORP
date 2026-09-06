"""
THE HARDWARE SEAM.

Everything above this interface -- Vision, Material, Classification,
Decision, Safety, Action -- is Phase-1 and Phase-2 identical. Everything
below it is either a simulation or an ESP32.

Two rules keep the seam honest, and both are cheap now and expensive later:

1. Every method returns an ActionResult carrying a status and a latency.
   The simulation cannot fail, so it is tempting to return None and assume
   success. Do that and the retry/hold logic never gets written, and Phase 2
   has to retrofit a failure path through working code -- exactly the
   redesign this architecture exists to avoid.

2. No agent may import a controller implementation. Agents take an
   ActionController; `build_controller` decides which one they get, from
   one word in config.yaml.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.config import Config
from core.messages import ActionResult, Destination

log = logging.getLogger(__name__)


class ActionController(ABC):
    """The only vocabulary the Action Agent knows."""

    name: str = "abstract"

    @abstractmethod
    def sort_to(self, destination: Destination, item_id: int) -> ActionResult:
        """Divert the item currently at the sorting point into `destination`."""

    @abstractmethod
    def start_conveyor(self) -> ActionResult: ...

    @abstractmethod
    def stop_conveyor(self, reason: str = "") -> ActionResult: ...

    @abstractmethod
    def hold_item(self, item_id: int, reason: str = "") -> ActionResult: ...

    @abstractmethod
    def check_status(self) -> dict:
        """Health of the actuation layer. Never raises -- reports instead."""

    @property
    def available(self) -> bool:
        return bool(self.check_status().get("available", False))


class ControllerUnavailable(RuntimeError):
    """Raised only at construction. Once built, a controller reports rather
    than raises, so a loose USB cable stops the belt instead of the process."""


def build_controller(cfg: Config, simulation=None) -> ActionController:
    """Pick the controller named in config.

    This function is the entire Phase 1 -> Phase 2 migration for the action
    layer. Nothing else in the codebase branches on which phase we are in.
    """
    kind = str(cfg.get("action.controller", "simulation")).lower()

    if kind == "simulation":
        from hardware.simulation_controller import SimulationController
        if simulation is None:
            raise ControllerUnavailable(
                "action.controller is 'simulation' but no simulation was "
                "supplied to build_controller()."
            )
        return SimulationController(cfg, simulation)

    if kind == "hardware":
        from hardware.hardware_controller import HardwareController
        return HardwareController(cfg)

    raise ControllerUnavailable(
        f"Unknown action.controller {kind!r} in {cfg.source}. "
        f"Expected 'simulation' (Phase 1) or 'hardware' (Phase 2)."
    )
