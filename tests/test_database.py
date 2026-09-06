"""
Stage 3 tests: persistence and the Monitoring Agent.

Every test writes to a throwaway database under the system temp directory, so
running these never touches data/waste.db.

Run:  python -m tests.test_database
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import tempfile

from core.config import Config
from core.event_bus import EventBus
from core.messages import Destination, Detection, Event, Severity
from database.database import Database
from hardware.simulation_controller import SimulationController
from pipeline import Pipeline
from simulation.world import World
from tests.test_pipeline import PASSED, FAILED, ScriptedDetector, box_at, check

logging.disable(logging.CRITICAL)


def temp_config(name: str) -> Config:
    cfg = Config.load()
    path = Path(tempfile.gettempdir()) / "waste_tests" / f"{name}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    cfg._data.setdefault("database", {})["path"] = str(path)
    # Sample and beat on every tick, so the tests do not wait ten seconds.
    cfg._data["database"]["bin_sample_seconds"] = 0
    cfg._data.setdefault("monitoring", {})["heartbeat_seconds"] = 0
    return cfg


def moving(track_id: int, label: str, confidence: float, count: int = 20):
    return [[Detection(track_id, label, confidence, box_at(300 + i * 30), i + 1)]
            for i in range(count)]


def build(cfg: Config, frames, db=None):
    bus = EventBus()
    world = World(cfg)
    controller = SimulationController(cfg, world)
    pipe = Pipeline(cfg, bus, world, controller,
                    detector=ScriptedDetector(frames), db=db)
    return bus, world, controller, pipe


def run(pipe, count, width=960):
    for frame_id in range(1, count + 1):
        pipe.process_frame(None, frame_id, width)


# ---------------------------------------------------------------------------

def test_schema_is_created():
    cfg = temp_config("schema")
    db = Database(cfg)
    tables = {r["name"] for r in db.query(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    check("all four tables exist",
          {"sessions", "detections", "events", "bin_status"} <= tables,
          f"tables={sorted(tables)}")
    mode = db.query("PRAGMA journal_mode")
    check("WAL is on so the report view can read while the line writes",
          mode and mode[0][0].lower() == "wal", f"mode={mode[0][0] if mode else None}")
    db.close()


def test_detections_are_persisted():
    cfg = temp_config("detections")
    db = Database(cfg)
    db.start_session("simulation", "test")
    _, _, _, pipe = build(cfg, moving(1, "banana", 0.95), db)
    run(pipe, 20)

    rows = db.recent_detections()
    check("one row per decided item", len(rows) == 1, f"rows={len(rows)}")
    if rows:
        row = rows[0]
        check("the row carries object, material, category and rule",
              row["object"] == "banana" and row["category"] == "ORGANIC"
              and row["safety_rule"] == "R7_APPROVED",
              f"{row['object']}/{row['category']}/{row['safety_rule']}")
        check("and the latency the actuator reported",
              row["latency_ms"] is not None)
    db.close()


def test_events_are_filtered():
    """Per-frame chatter belongs in the JSONL log, not in the database."""
    cfg = temp_config("events")
    db = Database(cfg)
    db.start_session("simulation", "test")
    bus = EventBus()
    bus.subscribe("*", db.insert_event)

    bus.publish(Event("BIN_ALERT", "MonitoringAgent",
                      {"bin": "ORGANIC", "reason": "nearing capacity"},
                      Severity.WARNING))
    bus.publish(Event("WASTE_DETECTED", "VisionAgent", {"label": "bottle"}))

    kept = {r["event"] for r in db.query("SELECT event FROM events")}
    check("alerts are kept", "BIN_ALERT" in kept)
    check("per-frame detections are not", "WASTE_DETECTED" not in kept,
          f"kept={sorted(kept)}")
    db.close()


def test_bin_snapshots_are_sampled():
    cfg = temp_config("bins")
    db = Database(cfg)
    db.start_session("simulation", "test")
    _, world, _, pipe = build(cfg, moving(2, "banana", 0.95), db)
    run(pipe, 20)
    pipe.monitoring.tick(now=100.0)

    rows = db.bin_history()
    check("every bin is sampled", len(rows) == len(world.bins.bins),
          f"rows={len(rows)} bins={len(world.bins.bins)}")
    db.close()


def test_aggregate_queries():
    cfg = temp_config("aggregates")
    db = Database(cfg)
    db.start_session("simulation", "test")

    for i, (label, conf) in enumerate(
            [("banana", 0.95), ("bottle", 0.94), ("cup", 0.93)]):
        _, _, _, pipe = build(cfg, moving(10 + i, label, conf), db)
        run(pipe, 20)

    summary = db.summary()
    check("summary counts every decided item",
          summary["processed"] == 3, f"processed={summary['processed']}")

    categories = {r["category"]: r["count"] for r in db.totals_by_category()}
    check("totals group by waste stream",
          categories.get("ORGANIC") == 1 and categories.get("RECYCLABLE") == 1
          and categories.get("MANUAL_CHECK") == 1, f"categories={categories}")

    rules = {r["safety_rule"]: r["count"] for r in db.totals_by_rule()}
    check("and by the safety rung that produced them",
          rules.get("R7_APPROVED") == 2 and rules.get("R4_UNCERTAIN") == 1,
          f"rules={rules}")

    days = db.daily_totals()
    check("daily totals are available for the report view", len(days) == 1,
          f"days={len(days)}")
    db.close()


def test_database_failure_does_not_stop_sorting():
    """The rule that matters most in this file.

    Recording that an item was sorted is bookkeeping. Sorting it is the job.
    A system that stops sorting because a file is locked has its priorities
    backwards, so the belt must keep running with persistence switched off.
    """
    cfg = temp_config("failure")
    db = Database(cfg)
    db.start_session("simulation", "test")
    _, world, controller, pipe = build(cfg, moving(3, "banana", 0.95), db)

    db.conn.close()                      # the disk goes away mid-shift

    run(pipe, 20)
    for _ in range(60):
        world.update(0.1)

    check("persistence switches itself off after the failure",
          not db.available)
    check("the item is still decided",
          pipe.stats.processed == 1, f"processed={pipe.stats.processed}")
    check("the actuator is still commanded",
          any(c.startswith("sort_to") for c in controller.commands))
    check("and the item still reaches its bin",
          world.bins[Destination.ORGANIC].current_level == 1)


def test_bin_alert_fires_once_then_rearms():
    cfg = temp_config("alerts")
    bus, world, _, pipe = build(cfg, moving(4, "banana", 0.95), None)
    organic = world.bins[Destination.ORGANIC]
    organic.current_level = int(organic.capacity * 0.85)     # over warn_at 0.80

    for _ in range(5):
        pipe.monitoring.tick(now=1.0)

    alerts = bus.recent("BIN_ALERT", limit=50)
    check("a threshold crossing alerts once, not once per frame",
          len(alerts) == 1, f"alerts={len(alerts)}")

    organic.current_level = organic.capacity                 # now full
    pipe.monitoring.tick(now=2.0)
    alerts = bus.recent("BIN_ALERT", limit=50)
    check("going from nearing-capacity to full raises a second, louder alert",
          len(alerts) == 2 and alerts[-1].severity is Severity.ALERT,
          f"alerts={len(alerts)}")

    organic.empty()
    pipe.monitoring.tick(now=3.0)
    organic.current_level = int(organic.capacity * 0.85)
    pipe.monitoring.tick(now=4.0)
    check("emptying the bin re-arms its alert",
          len(bus.recent("BIN_ALERT", limit=50)) == 3)


def test_monitoring_owns_the_counters():
    """The pipeline no longer counts anything itself."""
    cfg = temp_config("counters")
    _, _, _, pipe = build(cfg, moving(5, "banana", 0.95), None)
    run(pipe, 20)
    check("stats read through to the Monitoring Agent",
          pipe.stats is pipe.monitoring.stats)
    check("which learned about the item from the bus, not from a call",
          pipe.monitoring.stats.processed == 1
          and pipe.monitoring.records[0].label == "banana")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\n{'-' * 60}\n{len(PASSED)} passed, {len(FAILED)} failed")
    for failure in FAILED:
        print(f"  FAILED: {failure}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
