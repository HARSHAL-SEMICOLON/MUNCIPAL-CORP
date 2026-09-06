"""
Virtual bins.

In Phase 2 these become real containers with ultrasonic level sensors. The
interface is written so that swap changes only where `current_level` comes
from -- a sensor read instead of a counter -- and nothing that consumes a
Bin has to be rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import Config
from core.messages import Destination


@dataclass
class Bin:
    destination: Destination
    capacity: int
    warn_at: float                 # fraction of capacity that raises a warning
    current_level: int = 0
    accepted: list[str] = field(default_factory=list)

    @property
    def fill_fraction(self) -> float:
        return self.current_level / self.capacity if self.capacity else 1.0

    @property
    def percent(self) -> int:
        return int(round(self.fill_fraction * 100))

    @property
    def is_full(self) -> bool:
        return self.current_level >= self.capacity

    @property
    def needs_attention(self) -> bool:
        return self.fill_fraction >= self.warn_at

    @property
    def status(self) -> str:
        if self.is_full:
            return "FULL"
        if self.needs_attention:
            return "NEARING CAPACITY"
        return "OK"

    def add(self, label: str) -> bool:
        """Returns False when the bin is full -- the caller must handle it.

        Silently overfilling would make the Safety Agent's bin-full rung
        untestable, so this refuses instead.
        """
        if self.is_full:
            return False
        self.current_level += 1
        self.accepted.append(label)
        return True

    def empty(self) -> None:
        self.current_level = 0
        self.accepted.clear()


class BinFarm:
    """All bins, built from config so the set is editable without code changes."""

    def __init__(self, cfg: Config):
        self.bins: dict[Destination, Bin] = {}
        for name, spec in cfg.section("bins").items():
            try:
                destination = Destination(name)
            except ValueError:
                # A bin named in config that the vocabulary does not know is a
                # config error worth failing loudly on, not silently dropping.
                raise ValueError(
                    f"config bins.{name} is not a known Destination. "
                    f"Valid names: {[d.value for d in Destination if d is not Destination.NONE]}"
                ) from None
            self.bins[destination] = Bin(
                destination=destination,
                capacity=int(spec.get("capacity", 100)),
                warn_at=float(spec.get("warn_at", 0.8)),
            )

    def __getitem__(self, destination: Destination) -> Bin:
        return self.bins[destination]

    def get(self, destination: Destination) -> Bin | None:
        return self.bins.get(destination)

    def is_full(self, destination: Destination) -> bool:
        bin_ = self.bins.get(destination)
        return bool(bin_ and bin_.is_full)

    def warnings(self) -> list[Bin]:
        return [b for b in self.bins.values() if b.needs_attention]

    def snapshot(self) -> dict[str, dict]:
        return {
            b.destination.value: {
                "level": b.current_level,
                "capacity": b.capacity,
                "percent": b.percent,
                "status": b.status,
            }
            for b in self.bins.values()
        }

    @property
    def total_sorted(self) -> int:
        return sum(b.current_level for b in self.bins.values())
