"""
The closed loop: OBSERVE -> UNDERSTAND -> REASON -> DECIDE -> ACT -> MONITOR.

The real job of this file is the decision line, and it is worth explaining
because it is the difference between an agentic system and a classifier with
extra files around it.

Detection runs on every frame -- perception is continuous. But a physical
item must be decided ONCE. Without that separation, a bottle sitting in view
for two seconds at 30 fps produces sixty "decisions": sixty database rows,
sixty entries in the day's statistics, and in Phase 2 sixty servo commands
for one bottle. The system would be extremely busy and completely wrong.

So each tracked object accumulates evidence as it travels, and the agent
pipeline fires exactly once, when the object crosses a line partway across
the frame. Before the line it is being watched; after the line it has been
committed and is ignored. `min_frames_before_commit` stops a single-frame
flicker from committing anything at all.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from core.config import Config
from core.event_bus import EventBus
from core.labels import load_label_map
from core.messages import (Action, Category, Destination, Detection, Event,
                           Severity, Topic, WasteRecord)
from hardware.controller import ActionController
from simulation.waste_objects import WasteItem
from simulation.world import World

from agents.action_agent import ActionAgent
from agents.classification_agent import ClassificationAgent
from agents.decision_agent import DecisionAgent
from agents.material_agent import MaterialAgent
from agents.monitoring_agent import MonitoringAgent
from agents.routing_agent import RoutingAgent
from agents.safety_agent import SafetyAgent, SafetyContext
from agents.vision_agent import VisionAgent

log = logging.getLogger("Pipeline")


class Pipeline:
    def __init__(self, cfg: Config, bus: EventBus, world: World,
                 controller: ActionController, detector=None, db=None):
        self.cfg = cfg
        self.bus = bus
        self.world = world
        self.controller = controller

        # One label map, shared. Loading it twice would let the Material and
        # Classification agents disagree about the same detector's vocabulary.
        labels = load_label_map(cfg)
        self.labels = labels

        # `detector` is injectable so the whole loop can be tested headless,
        # with scripted detections and no camera or model involved. `db` is
        # optional for the same reason: the tests run without touching a disk.
        self.vision = VisionAgent(cfg, bus, detector)
        self.material = MaterialAgent(bus, labels=labels)
        self.classifier = ClassificationAgent(bus, labels=labels)
        self.decider = DecisionAgent(cfg, bus)
        self.safety = SafetyAgent(cfg, bus)
        self.actor = ActionAgent(cfg, bus, controller)
        self.routing = RoutingAgent(cfg, bus)
        self.monitoring = MonitoringAgent(cfg, bus, world, db)

        self.line_position = float(cfg.get("decision_line.position", 0.55))
        self.min_frames = int(cfg.get("decision_line.min_frames_before_commit", 5))
        self.forget_after = int(cfg.get("decision_line.forget_after_frames", 90))

        self._frames_tracked: dict[int, int] = defaultdict(int)
        self._last_seen: dict[int, int] = {}
        self._committed: set[int] = set()

        self.last_record: WasteRecord | None = None
        self._intrusion_active = False

    # The counters live in the Monitoring Agent now. These read through so the
    # renderer and the tests do not need to know that they moved.
    @property
    def stats(self):
        return self.monitoring.stats

    @property
    def records(self) -> list[WasteRecord]:
        return self.monitoring.records

    # -- per frame ---------------------------------------------------------

    def process_frame(self, frame, frame_id: int, frame_width: int) -> list[Detection]:
        detections = self.vision.observe(frame, frame_id)

        for det in detections:
            if det.track_id < 0:
                continue
            self._frames_tracked[det.track_id] += 1
            self._last_seen[det.track_id] = frame_id

        # The scene interlock runs before anything else and can end the frame.
        if self._scene_is_unsafe(detections):
            return detections

        # Nothing reaches the sorting point on a stopped belt, so nothing is
        # committed while it is stopped. This is physics, not a rule bolted
        # on: objects still in view are simply committed after the restart.
        if not self.world.conveyor.running:
            return detections

        line_x = frame_width * self.line_position
        for det in detections:
            if self._ready_to_commit(det, line_x):
                self._committed.add(det.track_id)
                self._commit(det, detections)

        self._forget_stale(frame_id)
        return detections

    def _scene_is_unsafe(self, detections: list[Detection]) -> bool:
        """Stop the belt while a person or animal is in the sorting zone.

        Announced on the transition only. The condition is true on every
        frame someone stands there, and emitting it thirty times a second
        would bury every other event on the bus.

        The belt is not restarted automatically when the zone clears. A real
        interlock requires a deliberate reset, and an operator who stepped
        away expects the belt to still be stopped when they look back.

        Announcing and acting are deliberately separated. The announcement
        fires once, on the transition, because the condition is true on every
        frame someone stands there and emitting it thirty times a second
        would bury every other event on the bus. **The stop is not
        transition-only.** It re-asserts on any frame where the zone is
        occupied and the belt is somehow moving -- which is exactly what
        happens when an operator presses the restart key without noticing
        that the person is still there. Running the system found that gap:
        the belt restarted, and nothing stopped it again.
        """
        verdict = self.safety.watch_scene(detections)

        if verdict is None:
            if self._intrusion_active:
                self._intrusion_active = False
                self.bus.publish(Event(
                    Topic.SYSTEM_UPDATE, "SafetyAgent",
                    {"reason": "sorting zone clear; belt awaiting operator restart"},
                    Severity.WARNING,
                ))
            return False

        if not self._intrusion_active:
            self._intrusion_active = True
            self.safety.emit(Topic.SAFETY_VERDICT, {
                "approved": False,
                "action": verdict.action,
                "rule": verdict.rule,
                "reason": verdict.reason,
            }, verdict.severity)

        # Idempotent, and outside the transition guard on purpose. A stopped
        # belt costs nothing here; a moving one with a person over it is the
        # failure this rung exists to prevent.
        if self.world.conveyor.running:
            self.actor.execute(verdict, item_id=-1, label="scene")
        return True

    def _ready_to_commit(self, det: Detection, line_x: float) -> bool:
        if det.track_id < 0 or det.track_id in self._committed:
            return False
        if self._frames_tracked[det.track_id] < self.min_frames:
            return False
        return det.centre[0] >= line_x

    def committed(self, track_id: int) -> bool:
        """Used by the renderer to show which objects have been acted on."""
        return track_id in self._committed

    def _forget_stale(self, frame_id: int) -> None:
        """Release bookkeeping for tracks that have left the scene.

        Trackers reuse ids eventually, so holding every id forever would
        mean a fresh object one day arrives already marked as committed and
        is silently never sorted.
        """
        gone = [tid for tid, seen in self._last_seen.items()
                if frame_id - seen > self.forget_after]
        for tid in gone:
            self._last_seen.pop(tid, None)
            self._frames_tracked.pop(tid, None)
            self._committed.discard(tid)
            self.vision.forget(tid)

    @staticmethod
    def _effective_stream(category: Category, destination: Destination) -> Category:
        """Which downstream route this item actually takes.

        Normally the classification: four categories share the recycling bin
        and separate again downstream, so glass must not be routed as paper
        merely because they left the line in the same container.

        But when the Safety Agent overrode the outcome, the stream genuinely
        changed. A low-confidence banana held for inspection is not entering
        the composting chain -- it is waiting for a person.
        """
        if destination is Destination.MANUAL_CHECK:
            return Category.MANUAL_CHECK
        if destination is Destination.HAZARDOUS:
            return Category.HAZARDOUS
        return category

    # -- one item, once ----------------------------------------------------

    def _commit(self, det: Detection, neighbours: list[Detection]) -> None:
        material = self.material.identify(det)
        classification = self.classifier.classify(det, material)
        decision = self.decider.decide(det, material, classification)

        verdict = self.safety.review(det, classification, decision, SafetyContext(
            neighbours=neighbours,
            bins=self.world.bins,
            conveyor=self.world.conveyor,
            controller=self.controller,
        ))

        # A stop is about the machine, not about this item, so nothing is
        # placed on the belt for it.
        if verdict.action is Action.STOP_CONVEYOR:
            result = self.actor.execute(verdict, item_id=-1, label=det.label)
        else:
            item = WasteItem(
                label=det.label,
                category=classification.category,
                destination=verdict.destination,
                track_id=det.track_id,
                confidence=det.confidence,
            )
            self.world.conveyor.place(item)
            result = self.actor.execute(verdict, item.item_id, det.label)

        stream = self._effective_stream(classification.category, verdict.destination)
        route = self.routing.assign(stream, det.track_id)

        record = WasteRecord(
            track_id=det.track_id,
            label=det.label,
            detection_confidence=det.confidence,
            material=material.material,
            category=classification.category,
            action=verdict.action,
            destination=verdict.destination,
            status=result.status,
            reason=verdict.reason,
            safety_rule=verdict.rule,
            latency_ms=result.latency_ms,
            route=route.chain if route else "",
        )
        self.last_record = record

        # Published, not handed over. The pipeline does not know that anything
        # is counting or persisting these -- which is precisely why Stage 4's
        # Analytics and Planning agents can subscribe here without this file
        # changing at all.
        self.bus.publish(Event(Topic.WASTE_RECORDED, "Pipeline", {
            "record": record,
            "label": record.label,
            "category": record.category,
            "action": record.action,
            "destination": record.destination,
            "status": record.status,
            "confidence": round(record.detection_confidence, 3),
            "rule": record.safety_rule,
        }))

    # -- monitoring --------------------------------------------------------

    def snapshot(self) -> dict:
        return self.monitoring.snapshot()
