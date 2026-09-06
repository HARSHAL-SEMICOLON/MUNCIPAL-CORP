"""Common ground for every agent: a name, the bus, and a way to speak."""

from __future__ import annotations

import logging

from core.event_bus import EventBus
from core.messages import Event, Severity


class Agent:
    name: str = "Agent"

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.log = logging.getLogger(self.name)

    def emit(self, topic: str, payload: dict, severity: Severity = Severity.INFO) -> None:
        self.bus.publish(Event(topic, self.name, payload, severity))
