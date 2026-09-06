"""
Renders one dashboard frame to logs/dashboard_preview.png without a camera.

Useful for checking the layout on a machine with no webcam, and for putting a
picture of the running system into a report.

Run:  python -m tests.smoke_render
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import cv2
import numpy as np

from core.config import Config
from core.event_bus import EventBus
from core.messages import Destination, Detection
from hardware.simulation_controller import SimulationController
from pipeline import Pipeline
from simulation.renderer import Renderer
from simulation.world import World
from tests.test_pipeline import ScriptedDetector, box_at

logging.disable(logging.CRITICAL)


def fake_camera_frame(w=960, h=540):
    """A stand-in for a camera image: a belt lit from above."""
    frame = np.full((h, w, 3), (38, 36, 34), dtype=np.uint8)
    cv2.rectangle(frame, (0, 200), (w, 330), (58, 55, 52), -1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 200), (x + 16, 330), (48, 46, 43), 2)
    cv2.putText(frame, "SIMULATED CAMERA INPUT", (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 108, 104), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    cfg = Config.load()
    bus, world = EventBus(), World(cfg)
    controller = SimulationController(cfg, world)

    # A bottle crossing the line, plus two objects that will trip the
    # occlusion rung, so the preview shows a populated belt.
    frames = []
    for i in range(24):
        x = 250 + i * 26
        row = [Detection(1, "bottle", 0.96, box_at(x), i + 1)]
        if i > 6:
            row.append(Detection(2, "banana", 0.88, box_at(x - 220), i + 1))
        if i > 12:
            row.append(Detection(3, "cell phone", 0.91, box_at(x - 420), i + 1))
        frames.append(row)

    pipe = Pipeline(cfg, bus, world, controller, detector=ScriptedDetector(frames))

    frame = fake_camera_frame()
    detections = []
    for frame_id in range(1, len(frames) + 1):
        detections = pipe.process_frame(frame, frame_id, frame.shape[1])
        world.update(0.08)

    # Pre-fill a bin so the capacity warning is visible in the preview.
    recycling = world.bins[Destination.RECYCLING]
    recycling.current_level = int(recycling.capacity * 0.86)

    canvas = Renderer(cfg).draw(frame, detections, pipe, world)
    out = Path(__file__).resolve().parent.parent / "logs" / "dashboard_preview.png"
    cv2.imwrite(str(out), canvas)

    print(f"wrote {out}  ({canvas.shape[1]}x{canvas.shape[0]})")
    print(f"items decided: {pipe.stats.processed}  on belt: {len(world.conveyor.items)}")
    for rec in pipe.records:
        print(f"  {rec.label:<12} {rec.category.value:<13} "
              f"{rec.action.value:<13} {rec.safety_rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
