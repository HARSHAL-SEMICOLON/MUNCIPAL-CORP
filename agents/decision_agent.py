"""
AGENT 4 -- DECISION AGENT

The central reasoning agent. It gathers what the perception agents concluded
and works out what ought to happen.

It PROPOSES. It does not command.

Nothing in this system acts on a Decision directly: it goes to the Safety
Agent, which holds the only path to the Action Agent. That separation is the
single most important structural choice in the architecture, and it is worth
being deliberate about why. An agent that both decides and acts has no place
to put a rule like "never sort an occluded object", because by the time you
know the object was occluded the servo has already moved. Splitting the two
gives every action a checkpoint it must pass, and gives every override a
name that can be printed in a log and defended in a review.

Sorting is what happens when nothing else applies -- the last resort, not
the default.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.messages import (Action, Category, Classification, Decision, Destination,
                           Detection, MaterialResult, Topic)
from core.routing import destination_for

from agents.base import Agent
from core.messages import Material


def _material_phrase(material: MaterialResult, category) -> str:
    """Read the material back the way a person would say it.

    Two small courtesies to whoever reads the dashboard. "UNKNOWN" is
    accurate but unhelpful when the Material Agent narrowed it to two
    possibilities and the Classification Agent established that both lead to
    the same place -- so name the possibilities. And when the material and
    the stream share a name, saying "ORGANIC (ORGANIC)" adds nothing.
    """
    if material.material is Material.UNKNOWN:
        if material.candidates:
            return " or ".join(c.value for c in material.candidates)
        return ""
    if material.material.value == category.value:
        return ""
    return material.material.value


class DecisionAgent(Agent):
    name = "DecisionAgent"

    def __init__(self, cfg: Config, bus: EventBus):
        super().__init__(bus)
        self.auto_sort = float(cfg.get("confidence.auto_sort", 0.85))
        self.verify = float(cfg.get("confidence.verify", 0.60))

    def decide(self, detection: Detection, material: MaterialResult,
               classification: Classification) -> Decision:
        confidence = detection.confidence
        category = classification.category

        if category is Category.MANUAL_CHECK:
            decision = Decision(
                Action.MANUAL_CHECK, Destination.MANUAL_CHECK, confidence,
                f"no stream could be assigned: {classification.reason}",
            )

        elif confidence < self.verify:
            decision = Decision(
                Action.MANUAL_CHECK, Destination.MANUAL_CHECK, confidence,
                f"detection confidence {confidence:.0%} is below the "
                f"{self.verify:.0%} inspection threshold",
            )

        elif confidence < self.auto_sort:
            # The middle band still sorts, but says out loud that it is not
            # certain, so the flag survives into the log and the dashboard.
            decision = Decision(
                Action.SORT, destination_for(category), confidence,
                f"{category.value} at {confidence:.0%} -- below the "
                f"{self.auto_sort:.0%} automatic threshold, sorting with the "
                f"decision flagged as uncertain",
            )

        else:
            phrase = _material_phrase(material, category)
            detail = f" ({phrase})" if phrase else ""
            decision = Decision(
                Action.SORT, destination_for(category), confidence,
                f"high-confidence {category.value}{detail} at {confidence:.0%}",
            )

        self.emit(Topic.SORT_DECISION, {
            "track_id": detection.track_id,
            "label": detection.label,
            "material": material.material,
            "category": category,
            "action": decision.action,
            "destination": decision.destination,
            "confidence": round(confidence, 3),
            "reason": decision.reason,
        })
        return decision
