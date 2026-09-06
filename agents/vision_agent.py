"""
AGENT 1 -- VISION AGENT

Perception only. It observes, it does not interpret: no waste category, no
material, no decision. Its single job is to turn frames into tracked
Detection objects and to announce each object once, when it first appears.

Announcing per object rather than per frame is deliberate. At 30 fps a
WASTE_DETECTED event per frame would put ~1800 events a minute on the bus
for a belt carrying a handful of items.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.messages import Detection, Topic
from vision.detector import Detector

from agents.base import Agent


class VisionAgent(Agent):
    name = "VisionAgent"

    def __init__(self, cfg: Config, bus: EventBus, detector: Detector | None = None):
        super().__init__(bus)
        self.detector = detector or Detector(cfg)
        self._announced: set[int] = set()
        self.frames_seen = 0
        self.detections_seen = 0

    def observe(self, frame, frame_id: int) -> list[Detection]:
        self.frames_seen += 1
        detections = self.detector.detect(frame, frame_id)
        self.detections_seen += len(detections)

        for det in detections:
            if det.track_id < 0 or det.track_id in self._announced:
                continue
            self._announced.add(det.track_id)
            self.emit(Topic.WASTE_DETECTED, {
                "track_id": det.track_id,
                "label": det.label,
                "confidence": round(det.confidence, 3),
                "bounding_box": list(det.bbox),
                "timestamp": det.timestamp,
            })
        return detections

    def forget(self, track_id: int) -> None:
        """Let a track id be announced again if the tracker reuses it."""
        self._announced.discard(track_id)
