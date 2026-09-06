"""One item travelling on the virtual belt."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

from core.messages import Category, Destination

_ids = itertools.count(1)


class ItemState(str, Enum):
    TRAVELLING = "TRAVELLING"   # moving toward its bin
    DELIVERED  = "DELIVERED"    # arrived, counted
    HELD       = "HELD"         # parked on the belt (bin full, retry pending)
    REJECTED   = "REJECTED"     # actuation failed permanently


# Drawn glyphs, chosen so a viewer can read the belt at a glance during a demo.
GLYPH = {
    Category.RECYCLABLE:   "PET",
    Category.GLASS:        "GLS",
    Category.METAL:        "MTL",
    Category.PAPER:        "PPR",
    Category.ORGANIC:      "ORG",
    Category.E_WASTE:      "EEE",
    Category.HAZARDOUS:    "HAZ",
    Category.REJECT:       "REJ",
    Category.MANUAL_CHECK: "?",
}


@dataclass
class WasteItem:
    label: str
    category: Category
    destination: Destination
    track_id: int
    confidence: float
    item_id: int = field(default_factory=lambda: next(_ids))
    progress: float = 0.0                  # 0.0 at the belt head, 1.0 at the bin
    state: ItemState = ItemState.TRAVELLING
    created_at: float = field(default_factory=time.monotonic)
    note: str = ""

    @property
    def glyph(self) -> str:
        return GLYPH.get(self.category, "?")

    def advance(self, dt: float, travel_seconds: float) -> None:
        """Move along the belt. A held item does not move."""
        if self.state is not ItemState.TRAVELLING or travel_seconds <= 0:
            return
        self.progress = min(1.0, self.progress + dt / travel_seconds)

    @property
    def arrived(self) -> bool:
        return self.state is ItemState.TRAVELLING and self.progress >= 1.0
