"""
AGENT 5 -- SAFETY AGENT

The gate. Every proposed decision passes through here, and nothing reaches
the Action Agent that has not been approved.

The rules are a ladder, checked hardest-consequence first, and the ordering
is the design. A hazardous item detected at 55% confidence must not be
handled as "low confidence, send to manual inspection" -- a person would
then be picking it up by hand. It is hazardous first and uncertain second,
so the hazard rung sits above the confidence rung.

Six of the seven rungs stop, hold or divert. Only an item that survives all
of them is sorted. That asymmetry is intentional: in a municipal setting the
cost of a wrong sort -- a contaminated recycling stream, a battery in
landfill, an injured worker -- is far higher than the cost of a held item.

One rule is about the scene rather than about any one item, and `watch_scene`
runs it on every frame instead of only when something is committed. Waiting
for a commit would mean a person could stand over a moving belt indefinitely,
as long as no waste happened to cross the line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import Config
from core.event_bus import EventBus
from core.messages import (Action, Category, Classification, Decision, Destination,
                           Detection, SafetyVerdict, Severity, Topic)

from agents.base import Agent


@dataclass
class SafetyContext:
    """Everything the Safety Agent needs that is not about this item alone."""
    neighbours: list[Detection] = field(default_factory=list)
    bins: object | None = None
    conveyor: object | None = None
    controller: object | None = None


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if overlap == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - overlap
    return overlap / union if union else 0.0


class SafetyAgent(Agent):
    name = "SafetyAgent"

    def __init__(self, cfg: Config, bus: EventBus):
        super().__init__(bus)
        self.verify = float(cfg.get("confidence.verify", 0.60))
        self.overlap_iou = float(cfg.get("safety.overlap_iou", 0.35))
        self.always_review = {
            Category(name) for name in cfg.get("safety.always_review", ["HAZARDOUS"])
        }
        self.intrusion_classes = {
            name.lower() for name in cfg.get("safety.intrusion_classes", ["person"])
        }

    def watch_scene(self, detections: list[Detection]) -> SafetyVerdict | None:
        """Scene-level interlock, evaluated every frame.

        Returns a verdict when the belt must stop, or None when the zone is
        clear. It does not emit -- the caller decides, because announcing an
        unchanged condition thirty times a second would bury every other
        event on the bus.
        """
        intruders = sorted({d.label for d in detections
                            if d.label.lower() in self.intrusion_classes})
        if not intruders:
            return None
        return SafetyVerdict(
            False, Action.STOP_CONVEYOR, Destination.NONE,
            f"{', '.join(intruders)} in the sorting zone; belt stopped and "
            f"held until an operator restarts it",
            rule="R2_INTRUSION", severity=Severity.CRITICAL,
        )

    def review(self, detection: Detection, classification: Classification,
               decision: Decision, context: SafetyContext) -> SafetyVerdict:
        verdict = self._apply_rules(detection, classification, decision, context)
        self.emit(Topic.SAFETY_VERDICT, {
            "track_id": detection.track_id,
            "label": detection.label,
            "approved": verdict.approved,
            "action": verdict.action,
            "destination": verdict.destination,
            "rule": verdict.rule,
            "reason": verdict.reason,
        }, verdict.severity)
        return verdict

    # -- the ladder --------------------------------------------------------

    def _apply_rules(self, detection: Detection, classification: Classification,
                     decision: Decision, ctx: SafetyContext) -> SafetyVerdict:

        # RUNG 1 -- the machine itself. Nothing else matters if the actuator
        # is not there to obey; keeping the belt running would feed items
        # past a diverter that cannot move.
        if ctx.controller is not None and not ctx.controller.available:
            return SafetyVerdict(
                False, Action.STOP_CONVEYOR, Destination.NONE,
                "actuation layer is unavailable; belt stopped rather than "
                "moving items past a diverter that cannot respond",
                rule="R1_CONTROLLER_UNAVAILABLE", severity=Severity.CRITICAL,
            )

        # RUNG 3 -- hazard outranks uncertainty. Checked before confidence so
        # a half-recognised hazardous item is never sent to a manual station.
        if classification.category in self.always_review:
            return SafetyVerdict(
                False, Action.SORT, Destination.HAZARDOUS,
                f"{classification.category.value} stream: diverted to the "
                f"hazardous route and flagged, regardless of confidence",
                rule="R3_HAZARDOUS", severity=Severity.ALERT,
            )

        # RUNG 4 -- uncertainty. Covers both a low score and a classification
        # that could not name a stream at all.
        if (detection.confidence < self.verify
                or classification.category is Category.MANUAL_CHECK):
            reason = (decision.reason if decision.action is Action.MANUAL_CHECK
                      else f"confidence {detection.confidence:.0%} below the "
                           f"{self.verify:.0%} inspection threshold")
            return SafetyVerdict(
                False, Action.MANUAL_CHECK, Destination.MANUAL_CHECK, reason,
                rule="R4_UNCERTAIN", severity=Severity.WARNING,
            )

        # RUNG 5 -- occlusion. Two heavily overlapping boxes are one
        # unresolved scene, not two confident detections, and that is the
        # classic route to a confident wrong answer.
        overlapping = [n for n in ctx.neighbours
                       if n.track_id != detection.track_id
                       and iou(n.bbox, detection.bbox) >= self.overlap_iou]
        if overlapping:
            others = ", ".join(sorted({n.label for n in overlapping}))
            return SafetyVerdict(
                False, Action.MANUAL_CHECK, Destination.MANUAL_CHECK,
                f"overlaps with {others}; an occluded object cannot be "
                f"classified reliably",
                rule="R5_OVERLAP", severity=Severity.WARNING,
            )

        # RUNG 6 -- capacity, checked before acting so the item is held on the
        # belt rather than arriving at a bin with nowhere to go.
        if ctx.bins is not None and ctx.bins.is_full(decision.destination):
            return SafetyVerdict(
                False, Action.HOLD, decision.destination,
                f"{decision.destination.value} bin is full; item held and "
                f"operator alerted",
                rule="R6_BIN_FULL", severity=Severity.ALERT,
            )

        # RUNG 7 -- nothing objected. Only now may the item be sorted.
        return SafetyVerdict(
            True, decision.action, decision.destination,
            decision.reason, rule="R7_APPROVED", severity=Severity.INFO,
        )
