"""
AGENT 6 -- MUNICIPAL ROUTING AGENT

Answers the question the bin cannot: what happens to this after collection?

The bin is the end of *this machine's* responsibility and the beginning of the
municipality's. Four different waste streams share the recycling bin, and they
diverge again downstream -- glass to glass recovery, fibre to paper recycling,
metal to ferrous/non-ferrous separation. Keeping the category alongside the
destination is what lets the system report "3 kg of glass" rather than only
"3 kg of recyclables", and this agent is where that distinction finally pays
for itself.

**On not inventing facilities.** Every route ships with `facility: null`. The
agent describes the *kind* of downstream destination -- "authorised e-waste
recycler" -- and will not name a specific one, because naming an organisation
that has not agreed to accept a stream is inventing a fact about a third
party. A report saying "batteries go to <company>" is worse than one saying
"batteries go to an authorised e-waste recycler": the first can be acted on
and be wrong. Fill in config/routing.yaml with verified local data and the
agent will report exactly what you put there, and nothing more.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from core.config import Config
from core.event_bus import EventBus
from core.messages import Category, Route, Topic

from agents.base import Agent

log = logging.getLogger("RoutingAgent")


class RoutingAgent(Agent):
    name = "RoutingAgent"

    def __init__(self, cfg: Config, bus: EventBus):
        super().__init__(bus)
        self.routes: dict[Category, Route] = {}
        self.source: Path | None = None
        self._load(cfg)

    def _load(self, cfg: Config) -> None:
        path = cfg.path("routing.file", "config/routing.yaml")
        self.source = path
        if not path.exists():
            # Not fatal. A missing routing table means the system cannot say
            # what happens downstream -- it does not mean it should stop
            # sorting, or start guessing.
            log.warning("No routing table at %s; downstream routes unavailable", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError:
            log.exception("Could not parse %s; downstream routes unavailable", path)
            return

        for name, spec in (data.get("routes") or {}).items():
            try:
                category = Category(name)
            except ValueError:
                log.warning("routing.yaml names an unknown category %r; ignoring", name)
                continue
            self.routes[category] = Route(
                stream=str(spec.get("stream", "")).strip(),
                stages=tuple(str(s).strip() for s in (spec.get("stages") or [])),
                facility=spec.get("facility") or None,
                notes=" ".join(str(spec.get("notes", "")).split()),
            )

        missing = [c.value for c in Category if c not in self.routes]
        if missing:
            log.warning("routing.yaml has no route for: %s", ", ".join(missing))
        log.info("Routing table loaded: %d of %d categories",
                 len(self.routes), len(list(Category)))

    def route_for(self, category: Category) -> Route | None:
        return self.routes.get(category)

    def assign(self, category: Category, track_id: int = -1) -> Route | None:
        route = self.routes.get(category)
        if route is None:
            self.emit(Topic.ROUTE_ASSIGNED, {
                "track_id": track_id,
                "category": category,
                "route": "",
                "reason": f"no downstream route configured for {category.value}",
            })
            return None

        self.emit(Topic.ROUTE_ASSIGNED, {
            "track_id": track_id,
            "category": category,
            "stream": route.stream,
            "route": route.chain,
            # Reported only when a municipality has supplied it; never guessed.
            "facility": route.facility,
        })
        return route

    def table(self) -> list[dict]:
        """The routing table, for the reporting view."""
        return [
            {
                "category": category.value,
                "stream": route.stream,
                "chain": route.chain,
                "facility": route.facility or "-- not configured --",
                "notes": route.notes,
            }
            for category, route in self.routes.items()
        ]

    @property
    def configured_facilities(self) -> int:
        return sum(1 for r in self.routes.values() if r.facility)
