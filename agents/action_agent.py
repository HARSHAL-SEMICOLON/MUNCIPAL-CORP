"""
AGENT 7 -- ACTION AGENT

Carries out an approved verdict, and is the last piece of code that is
identical in Phase 1 and Phase 2.

It talks only to an ActionController. It has never heard of a conveyor, a
servo or a serial port. Swap SimulationController for HardwareController and
this file does not change -- that is the whole return on building the seam.

The retry-then-hold logic below is written now, against a simulation that
cannot fail, because the config flag `action.simulated_failure_rate` can make
it fail on demand. Writing it later, against real hardware, would mean
threading a failure path through code that already works -- which is exactly
how a project ends up needing the redesign this architecture exists to avoid.

Note what is NOT retried. A REFUSED result means the controller understood
the command and declined it: the bin is full, or the belt is stopped.
Retrying that is pointless and would just delay the operator alert. Only
FAILED and TIMEOUT -- an actuator that tried and did not confirm -- are worth
a second attempt.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.messages import (Action, ActionResult, ActionStatus, Destination,
                           SafetyVerdict, Severity, Topic)
from hardware.controller import ActionController

from agents.base import Agent

RETRYABLE = (ActionStatus.FAILED, ActionStatus.TIMEOUT)


class ActionAgent(Agent):
    name = "ActionAgent"

    def __init__(self, cfg: Config, bus: EventBus, controller: ActionController):
        super().__init__(bus)
        self.controller = controller
        self.max_retries = int(cfg.get("action.max_retries", 2))
        self.executed = 0
        self.failed = 0

    def execute(self, verdict: SafetyVerdict, item_id: int, label: str = "") -> ActionResult:
        if verdict.action is Action.STOP_CONVEYOR:
            result = self.controller.stop_conveyor(verdict.reason)
        elif verdict.action is Action.HOLD:
            result = self.controller.hold_item(item_id, verdict.reason)
        elif verdict.action in (Action.SORT, Action.MANUAL_CHECK, Action.REJECT):
            result = self._sort_with_retry(verdict.destination, item_id)
        else:
            result = ActionResult(
                ActionStatus.REFUSED, Destination.NONE, 0.0,
                detail=f"no actuation defined for {verdict.action.value}",
            )

        self.executed += 1
        if not result.ok:
            self.failed += 1

        self.emit(Topic.SORT_EXECUTED, {
            "item_id": item_id,
            "label": label,
            "action": verdict.action,
            "destination": result.destination,
            "status": result.status,
            "attempts": result.attempts,
            "latency_ms": round(result.latency_ms, 2),
            "detail": result.detail,
        }, Severity.INFO if result.ok else Severity.ALERT)
        return result

    def _sort_with_retry(self, destination: Destination, item_id: int) -> ActionResult:
        attempts = 0
        total_latency = 0.0
        result = None

        while attempts <= self.max_retries:
            attempts += 1
            result = self.controller.sort_to(destination, item_id)
            total_latency += result.latency_ms
            if result.ok or result.status not in RETRYABLE:
                break
            self.log.warning(
                "sort_to(%s) attempt %d/%d returned %s: %s",
                destination.value, attempts, self.max_retries + 1,
                result.status.value, result.detail,
            )

        result.attempts = attempts
        result.latency_ms = total_latency

        if not result.ok:
            # Out of attempts. Park the item rather than let it ride the belt
            # to an unknown end -- an unsorted item on the floor is worse than
            # a held one an operator can see.
            hold = self.controller.hold_item(
                item_id, f"actuation {result.status.value} after {attempts} attempts")
            result.detail = (
                f"{result.detail}; held on belt" if hold.ok
                else f"{result.detail}; hold also failed ({hold.detail})"
            )
        return result
