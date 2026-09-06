"""
YOLO detection with persistent tracking.

Tracking is not a nicety here, it is what makes the agent pipeline correct.
Detection runs every frame; without a stable identity per object, a bottle
sitting in view for two seconds produces sixty independent "decisions" --
sixty database rows, and in Phase 2 sixty servo commands for one bottle.

`model.track(persist=True)` gives every object a track_id that survives
across frames, so the pipeline can commit to each physical item exactly once.
See the decision-line logic in pipeline.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.config import Config
from core.messages import Detection

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class Detector:
    def __init__(self, cfg: Config):
        model_path: Path = cfg.path("detection.model_path", "models/yolov8n.pt")
        if not model_path.exists():
            raise ModelUnavailable(
                f"No detection model at {model_path}. Download yolov8n.pt "
                f"(ultralytics will fetch it on first use) and place it there."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailable(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        self.model = YOLO(str(model_path))
        self.imgsz = int(cfg.get("detection.imgsz", 640))
        self.device = str(cfg.get("detection.device", "cpu"))
        self.min_confidence = float(cfg.get("detection.min_confidence", 0.35))
        self.tracker = str(cfg.get("detection.tracker", "bytetrack.yaml"))
        self.names: dict[int, str] = self.model.names
        log.info("Detector ready: %s on %s (%d classes)",
                 model_path.name, self.device, len(self.names))

    def detect(self, frame, frame_id: int) -> list[Detection]:
        """One frame in, tracked detections out. Never raises on a bad frame."""
        try:
            results = self.model.track(
                frame,
                persist=True,               # keeps track ids across calls
                tracker=self.tracker,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.min_confidence,
                verbose=False,
            )
        except Exception:
            # A single bad frame must not end the run; the belt keeps moving
            # and the next frame gets another chance.
            log.exception("Detection failed on frame %d; skipping", frame_id)
            return []

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for i in range(len(boxes)):
            confidence = float(boxes.conf[i])
            if confidence < self.min_confidence:
                continue

            # Without an assigned track id the object cannot be committed
            # exactly once, so it is observed but not acted on.
            if boxes.id is None:
                track_id = -1
            else:
                track_id = int(boxes.id[i])

            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
            label = self.names.get(int(boxes.cls[i]), "unknown")
            detections.append(Detection(
                track_id=track_id,
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                frame_id=frame_id,
            ))
        return detections
