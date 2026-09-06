"""The simulated plant: bins plus the belt that feeds them.

Exists so the SimulationController has one object to talk to, and so Phase 2
has one object to stop constructing.
"""

from __future__ import annotations

from core.config import Config
from simulation.bins import BinFarm
from simulation.conveyor import Conveyor


class World:
    def __init__(self, cfg: Config):
        self.bins = BinFarm(cfg)
        self.conveyor = Conveyor(cfg, self.bins)

    def update(self, dt: float):
        return self.conveyor.update(dt)

    def snapshot(self) -> dict:
        return {"conveyor": self.conveyor.snapshot(), "bins": self.bins.snapshot()}
