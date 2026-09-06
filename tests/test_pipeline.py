"""
Headless tests for the whole agent loop -- no camera, no model, no window.

The detector is replaced with a scripted one, so every test below is about
the agents and their contracts rather than about YOLO. That is the payoff of
keeping perception behind an interface: the reasoning can be tested on a
laptop with no webcam, in a fraction of a second, deterministically.

Run:  python -m tests.test_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from core.config import Config
from core.event_bus import EventBus
from core.messages import (Action, ActionStatus, Category, Destination, Detection,
                           Material, Topic)
from hardware.simulation_controller import SimulationController
from pipeline import Pipeline
from simulation.world import World

logging.disable(logging.CRITICAL)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(f"{name}  {detail}".rstrip())
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


class ScriptedDetector:
    """Returns a fixed list of detections for each frame, in order."""

    def __init__(self, frames: list[list[Detection]]):
        self.frames = frames
        self.names = {0: "scripted"}

    def detect(self, frame, frame_id: int) -> list[Detection]:
        index = frame_id - 1
        return self.frames[index] if 0 <= index < len(self.frames) else []


def box_at(x: int) -> tuple[int, int, int, int]:
    return (x - 40, 200, x + 40, 300)


def build(cfg: Config, frames: list[list[Detection]]):
    bus = EventBus()
    world = World(cfg)
    controller = SimulationController(cfg, world)
    pipe = Pipeline(cfg, bus, world, controller, detector=ScriptedDetector(frames))
    return bus, world, controller, pipe


def run_frames(pipe: Pipeline, count: int, width: int = 960) -> None:
    for frame_id in range(1, count + 1):
        pipe.process_frame(None, frame_id, width)


# ---------------------------------------------------------------------------

def test_one_decision_per_object():
    """The defect this architecture exists to prevent.

    A bottle visible for 40 frames must produce exactly one decision, not 40.
    """
    cfg = Config.load()
    # Same track id every frame, drifting across the decision line at 0.55.
    frames = [[Detection(7, "bottle", 0.94, box_at(200 + i * 20), i + 1)]
              for i in range(40)]
    bus, world, controller, pipe = build(cfg, frames)
    run_frames(pipe, 40)

    check("one decision per tracked object",
          pipe.stats.processed == 1, f"processed={pipe.stats.processed}")
    check("one actuator command per object",
          len([c for c in controller.commands if c.startswith("sort_to")]) == 1,
          f"sort_to calls={[c for c in controller.commands if c.startswith('sort_to')]}")
    check("decision events not emitted per frame",
          len(bus.recent(Topic.SORT_DECISION, limit=100)) == 1)


def test_waits_for_the_line():
    """An object that never crosses the line is watched, never committed."""
    cfg = Config.load()
    frames = [[Detection(3, "bottle", 0.94, box_at(120), i + 1)] for i in range(30)]
    _, _, _, pipe = build(cfg, frames)
    run_frames(pipe, 30)
    check("object short of the line is not committed",
          pipe.stats.processed == 0, f"processed={pipe.stats.processed}")


def test_flicker_is_ignored():
    """A single-frame detection past the line must not commit anything."""
    cfg = Config.load()
    frames = [[]] * 5 + [[Detection(11, "bottle", 0.94, box_at(800), 6)]] + [[]] * 5
    _, _, _, pipe = build(cfg, frames)
    run_frames(pipe, 11)
    check("single-frame flicker is not committed",
          pipe.stats.processed == 0, f"processed={pipe.stats.processed}")


def test_bottle_ambiguity_still_sorts():
    """PET or glass -- same bin either way, so the item proceeds."""
    cfg = Config.load()
    frames = [[Detection(21, "bottle", 0.96, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, world, _, pipe = build(cfg, frames)
    run_frames(pipe, 20)
    rec = pipe.last_record
    check("ambiguous bottle is not sent to a human",
          rec is not None and rec.action is Action.SORT, f"action={rec.action.value}")
    check("material honestly reported as UNKNOWN",
          rec.material is Material.UNKNOWN)
    check("category generalised to RECYCLABLE",
          rec.category is Category.RECYCLABLE, f"category={rec.category.value}")
    check("routed to the recycling bin",
          rec.destination is Destination.RECYCLING)


def test_cup_ambiguity_defers():
    """Plastic, paper or ceramic -- different bins, so a human decides."""
    cfg = Config.load()
    frames = [[Detection(22, "cup", 0.96, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, _, _, pipe = build(cfg, frames)
    run_frames(pipe, 20)
    rec = pipe.last_record
    check("divergent ambiguity goes to manual check",
          rec.action is Action.MANUAL_CHECK, f"action={rec.action.value}")
    check("and is attributed to the uncertainty rung",
          rec.safety_rule == "R4_UNCERTAIN", f"rule={rec.safety_rule}")


def test_low_confidence_defers():
    cfg = Config.load()
    frames = [[Detection(31, "banana", 0.41, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, _, _, pipe = build(cfg, frames)
    run_frames(pipe, 20)
    check("low confidence never sorts",
          pipe.last_record.action is Action.MANUAL_CHECK)


def test_hazard_outranks_uncertainty():
    """The rung ordering that matters most: a 52%-confidence knife must not
    be handed to a person as a routine manual inspection."""
    cfg = Config.load()
    frames = [[Detection(41, "knife", 0.52, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, _, _, pipe = build(cfg, frames)
    run_frames(pipe, 20)
    rec = pipe.last_record
    check("hazard rung fires before the confidence rung",
          rec.safety_rule == "R3_HAZARDOUS", f"rule={rec.safety_rule}")
    check("and diverts to the hazardous route",
          rec.destination is Destination.HAZARDOUS, f"dest={rec.destination.value}")


def test_overlap_defers():
    """Two heavily overlapping objects are one unresolved scene."""
    cfg = Config.load()
    frames = []
    for i in range(20):
        x = 300 + i * 30
        frames.append([
            Detection(51, "banana", 0.93, box_at(x), i + 1),
            Detection(52, "apple", 0.91, (x - 30, 210, x + 50, 310), i + 1),
        ])
    _, _, _, pipe = build(cfg, frames)
    run_frames(pipe, 20)
    rules = set(pipe.stats.by_rule)
    check("occluded objects go to manual check",
          rules == {"R5_OVERLAP"}, f"rules={rules}")


def test_full_bin_holds():
    cfg = Config.load()
    frames = [[Detection(61, "banana", 0.95, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, world, _, pipe = build(cfg, frames)
    organic = world.bins[Destination.ORGANIC]
    organic.current_level = organic.capacity          # fill it
    run_frames(pipe, 20)
    rec = pipe.last_record
    check("a full bin holds the item instead of sorting",
          rec.action is Action.HOLD and rec.safety_rule == "R6_BIN_FULL",
          f"action={rec.action.value} rule={rec.safety_rule}")


def test_actuator_failure_retries_then_holds():
    """The reason ActionResult exists from day one.

    The simulation cannot fail on its own, so failure is injected. This is
    the path that will meet a real servo in Phase 2, tested before one exists.
    """
    cfg = Config.load()
    frames = [[Detection(71, "banana", 0.95, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    bus, world, controller, pipe = build(cfg, frames)
    controller.failure_rate = 1.0                     # every command fails
    run_frames(pipe, 20)

    rec = pipe.last_record
    attempts = len([c for c in controller.commands if c.startswith("sort_to")])
    expected = int(cfg.get("action.max_retries", 2)) + 1
    check("a failing actuator is retried to the configured limit",
          attempts == expected, f"attempts={attempts} expected={expected}")
    check("and the item is then held, not lost",
          rec.status is not ActionStatus.OK
          and any(c.startswith("hold_item") for c in controller.commands),
          f"status={rec.status.value}")


def test_controller_unavailable_stops_the_belt():
    cfg = Config.load()
    frames = [[Detection(81, "banana", 0.95, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, world, controller, pipe = build(cfg, frames)
    controller.check_status = lambda: {"available": False, "controller": "test"}
    run_frames(pipe, 20)
    check("an unavailable actuator stops the conveyor",
          pipe.last_record.safety_rule == "R1_CONTROLLER_UNAVAILABLE"
          and not world.conveyor.running,
          f"rule={pipe.last_record.safety_rule} running={world.conveyor.running}")


def test_items_reach_their_bins():
    """End to end: detection through to a physical count in a bin."""
    cfg = Config.load()
    frames = [[Detection(91, "banana", 0.95, box_at(300 + i * 30), i + 1)]
              for i in range(20)]
    _, world, _, pipe = build(cfg, frames)
    run_frames(pipe, 20)
    for _ in range(60):                               # advance the belt 6s
        world.update(0.1)
    check("the item is delivered into the organic bin",
          world.bins[Destination.ORGANIC].current_level == 1,
          f"level={world.bins[Destination.ORGANIC].current_level}")


def test_person_in_zone_stops_the_belt():
    """The interlock. A person over a moving belt outranks every sorting rule,
    and the belt must not restart by itself when they step away."""
    cfg = Config.load()
    frames = []
    for i in range(24):
        row = [Detection(101, "bottle", 0.96, box_at(300 + i * 30), i + 1)]
        if i < 16:                                    # person leaves after frame 16
            row.append(Detection(102, "person", 0.88, (10, 10, 120, 400), i + 1))
        frames.append(row)

    bus, world, controller, pipe = build(cfg, frames)
    run_frames(pipe, 24)

    stops = [c for c in controller.commands if c.startswith("stop_conveyor")]
    check("a person in the zone stops the belt",
          len(stops) >= 1 and not world.conveyor.running, f"stops={len(stops)}")
    check("the stop is announced once, not every frame",
          len(stops) == 1, f"stops={len(stops)}")
    check("nothing is sorted while the zone is occupied",
          pipe.stats.processed == 0, f"processed={pipe.stats.processed}")
    check("the belt does not restart itself once the zone clears",
          not world.conveyor.running)


def test_operator_cannot_restart_into_an_occupied_zone():
    """Found by running the system, not by reading it.

    The stop fired on the transition into intrusion and never again. So an
    operator pressing the restart key while the person was still in frame got
    a moving belt, and nothing stopped it -- the interlock had already had
    its say. The announcement stays transition-only; the stop does not.
    """
    cfg = Config.load()
    frames = [[Detection(111, "person", 0.88, (10, 10, 120, 400), i + 1)]
              for i in range(30)]
    bus, world, controller, pipe = build(cfg, frames)

    run_frames(pipe, 6)
    check("the belt stops when the person appears", not world.conveyor.running)

    controller.start_conveyor()                 # the operator presses S
    check("the operator's restart does take effect", world.conveyor.running)

    run_frames(pipe, 6)
    check("but the next frame stops it again, person still present",
          not world.conveyor.running)

    verdicts = bus.recent(Topic.SAFETY_VERDICT, limit=50)
    check("and the alert is still announced only once, not once per frame",
          len(verdicts) == 1, f"announcements={len(verdicts)}")


def test_phase_two_seam_is_selectable():
    """The Phase 1 -> Phase 2 migration is one word in config.

    Since Stage 6, selecting `hardware` builds a real controller that tries
    to open a serial port, so this checks the selection and the honest
    refusal when no port is configured. The controller's own behaviour --
    against a simulated board -- is covered in tests/test_hardware.py.
    """
    from hardware.controller import ControllerUnavailable, build_controller
    from hardware.hardware_controller import HardwareController
    from hardware.fake_device import FakeDevice
    from hardware.transport import LoopbackTransport

    cfg = Config.load()
    world = World(cfg)
    sim = build_controller(cfg, world)
    check("config selects the simulation controller", sim.name == "simulation")

    cfg._data["action"]["controller"] = "hardware"
    cfg._data.setdefault("hardware", {})["port"] = None
    try:
        build_controller(cfg, world)
        check("selecting hardware with no port is refused", False,
              "it was accepted")
    except ControllerUnavailable as exc:
        check("selecting hardware with no port is refused, naming the key",
              "hardware.port" in str(exc), str(exc)[:60])
    finally:
        cfg._data["action"]["controller"] = "simulation"

    # Given a link, the same one-word switch yields a working controller.
    hw = HardwareController(cfg, transport=LoopbackTransport(
        FakeDevice(bin_angles={"ORGANIC": 60}, travel_ms=5.0)))
    check("and with a link it is a drop-in ActionController",
          isinstance(hw, HardwareController) and hw.available)


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
