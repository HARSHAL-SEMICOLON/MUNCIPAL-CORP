"""
Fill data/waste.db by running the real agents over scripted detections.

    python -m tests.seed_demo_data [--items 60] [--reset]

Nothing here fabricates a record. The detections stand in for a camera -- the
same substitution the tests make -- but every material identification, waste
category, safety verdict and actuator result is produced by the real agents at
the moment the row is written. What you see in the reporting view is what this
system actually decided.

The one liberty taken is the clock. A few minutes of belt time run in about a
second of real time, so bin samples carry the simulated clock, laid out so the
shift ends now; otherwise every sample would share one timestamp. The session
row records the source as scripted, so the run is never mistaken for a live
shift.

Useful for checking the reporting view without holding objects up to a webcam
for ten minutes.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from core.config import Config
from core.event_bus import EventBus
from core.messages import Detection
from database.database import Database
from hardware.simulation_controller import SimulationController
from pipeline import Pipeline
from simulation.world import World

logging.disable(logging.CRITICAL)

# A plausible mix for a municipal line: mostly organics and dry recyclables,
# a few electronics, the occasional sharp, and some objects the model has no
# material rule for at all.
MIX = [
    ("banana", 0.94, 14), ("apple", 0.91, 8), ("sandwich", 0.88, 5),
    ("bottle", 0.95, 16), ("book", 0.90, 6), ("wine glass", 0.87, 4),
    ("cup", 0.92, 6), ("cell phone", 0.93, 4), ("laptop", 0.89, 2),
    ("knife", 0.71, 2), ("scissors", 0.66, 1),
    ("umbrella", 0.85, 3), ("teddy bear", 0.62, 2),
    ("bottle", 0.48, 3), ("banana", 0.52, 2),      # low-confidence stragglers
]


class ScriptedDetector:
    def __init__(self, frames):
        self.frames = frames
        self.names = {0: "scripted"}

    def detect(self, frame, frame_id: int):
        index = frame_id - 1
        return self.frames[index] if 0 <= index < len(self.frames) else []


def build_frames(items: int, rng: random.Random):
    """One object at a time, each crossing the decision line over 14 frames."""
    plan = []
    for label, confidence, weight in MIX:
        plan.extend([(label, confidence)] * weight)
    rng.shuffle(plan)

    frames, track_id = [], 1
    while len(plan) and track_id <= items:
        label, base = plan[(track_id - 1) % len(plan)]
        confidence = max(0.30, min(0.99, base + rng.uniform(-0.05, 0.04)))
        for i in range(14):
            x = 260 + i * 34
            frames.append([Detection(track_id, label, confidence,
                                     (x - 40, 200, x + 40, 300), len(frames) + 1)])
        frames.extend([[]] * 3)          # a gap, so tracks do not run together
        track_id += 1
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=60)
    parser.add_argument("--reset", action="store_true",
                        help="delete the database first")
    args = parser.parse_args()

    cfg = Config.load()
    path = cfg.path("database.path", "data/waste.db")
    if args.reset:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                candidate.unlink()
        print(f"removed {path}")

    rng = random.Random(20260905)
    frames = build_frames(args.items, rng)

    db = Database(cfg)
    db.start_session("simulation", "scripted (tests.seed_demo_data)")

    bus = EventBus()
    world = World(cfg)
    controller = SimulationController(cfg, world)
    # A little injected unreliability, so the retry-then-hold path appears in
    # the record rather than being a branch nobody has ever seen fire.
    controller.failure_rate = 0.06
    pipe = Pipeline(cfg, bus, world, controller,
                    detector=ScriptedDetector(frames), db=db)

    # The run compresses a few minutes of belt time into about a second of
    # real time, so bin samples are stamped with the simulated clock, laid out
    # so the shift ends now. Without it every sample carries one timestamp and
    # the fill-over-time chart is a single point.
    total_seconds = len(frames) * 0.09 + 8.0
    started = datetime.now(timezone.utc) - timedelta(seconds=total_seconds)

    def stamp(elapsed: float) -> str:
        return (started + timedelta(seconds=elapsed)).isoformat(timespec="seconds")

    clock = 0.0
    for frame_id in range(1, len(frames) + 1):
        pipe.process_frame(None, frame_id, 960)
        world.update(0.09)
        clock += 0.09
        pipe.monitoring.tick(now=clock, at=stamp(clock))

    for _ in range(80):                  # let the belt finish delivering
        world.update(0.1)
        clock += 0.1

    db.snapshot_bins(world.bins, stamp(clock))
    db.end_session(len(frames))

    snap = pipe.snapshot()
    print(f"\nwrote {snap['processed']} decisions to {path}")
    print(f"  sorted {snap['sorted_ok']}   manual {snap['manual_checks']}   "
          f"failures {snap['failures']}   avg confidence "
          f"{snap['average_confidence']:.0%}")
    print("\n  by category:")
    for name, count in sorted(snap["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:<14} {count}")
    print("\n  by safety rule:")
    for name, count in sorted(snap["by_rule"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:<28} {count}")
    print(f"\n  bins: {world.bins.snapshot()}")
    print("\n  streamlit run dashboard/app.py")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
