"""
A synchronous publish/subscribe bus.

Deliberately small. Agents never import each other; they publish typed events
and whoever cares subscribes. That is the whole reason the Monitoring,
Analytics and Planning agents can be added in later stages without touching
the sorting path.

Synchronous by choice: the sorting line makes one decision per object at a few
objects per second, so a thread pool would add failure modes and buy nothing.
If Phase 2 needs asynchronous delivery, only this file changes.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Iterable

from core.messages import Event, Severity

log = logging.getLogger(__name__)

Handler = Callable[[Event], None]
WILDCARD = "*"


class EventBus:
    def __init__(self, history: int = 400):
        self._subscribers: dict[str, list[Handler]] = {}
        self.history: deque[Event] = deque(maxlen=history)

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe to one topic, or to WILDCARD for every event."""
        self._subscribers.setdefault(topic, []).append(handler)

    def subscribe_many(self, topics: Iterable[str], handler: Handler) -> None:
        for topic in topics:
            self.subscribe(topic, handler)

    def publish(self, event: Event) -> None:
        self.history.append(event)
        for handler in list(self._subscribers.get(event.event, ())):
            self._dispatch(handler, event)
        for handler in list(self._subscribers.get(WILDCARD, ())):
            self._dispatch(handler, event)

    @staticmethod
    def _dispatch(handler: Handler, event: Event) -> None:
        """A broken subscriber must never take the sorting line down.

        The dashboard, the event recorder and later the database all hang off
        this bus. If one of them raises, the item on the belt still gets
        sorted -- the failure is logged, not propagated.
        """
        try:
            handler(event)
        except Exception:
            log.exception(
                "Subscriber %r failed handling %s; continuing",
                getattr(handler, "__qualname__", handler), event.event,
            )

    def recent(self, topic: str | None = None, limit: int = 20) -> list[Event]:
        items = [e for e in self.history if topic is None or e.event == topic]
        return items[-limit:]

    def recent_alerts(self, limit: int = 5) -> list[Event]:
        loud = (Severity.WARNING, Severity.ALERT, Severity.CRITICAL)
        return [e for e in self.history if e.severity in loud][-limit:]
