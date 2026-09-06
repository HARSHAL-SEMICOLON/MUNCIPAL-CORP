"""
Phase 1 entry point.

    python main.py                       webcam, as configured
    python main.py --source data/x.mp4   a recorded clip instead
    python main.py --headless 300        300 frames, no window (CI / smoke test)

Keys while running:
    q   quit            s   start/stop the belt
    e   empty the bins  f   toggle injected actuator failures
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time

from core.config import Config
from core.event_bus import EventBus
from core.logging_setup import EventRecorder, setup_logging
from core.messages import Event, Severity, Topic
from database.database import Database
from hardware.controller import ControllerUnavailable, build_controller
from pipeline import Pipeline
from simulation.world import World
from vision.camera import Camera, CameraUnavailable
from vision.detector import ModelUnavailable

log = logging.getLogger("main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Smart Municipal Waste Segregation -- Phase 1")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--source", default=None,
                   help="override camera.source (webcam index or video file)")
    p.add_argument("--headless", type=int, default=0, metavar="N",
                   help="run N frames without a window, then print a summary")
    p.add_argument("--report", action="store_true",
                   help="print insights and recommendations from the record "
                        "and exit; no camera is opened")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = Config.load(args.config)
    if args.source is not None:
        cfg._data.setdefault("camera", {})["source"] = args.source

    setup_logging(cfg)
    bus = EventBus()

    if args.report:
        # Reporting reads the record and opens nothing else. No camera, no
        # model, no belt -- so it works on a machine that has never had a
        # webcam attached.
        db = Database(cfg)
        code = print_report(cfg, bus, db)
        db.close()
        return code

    recorder = EventRecorder(cfg, bus)

    # Every start-up failure below is reported plainly and exits, rather than
    # surfacing as a stack trace three layers into OpenCV. A system that
    # cannot see must not pretend to sort.
    try:
        camera = Camera(cfg)
    except (CameraUnavailable, FileNotFoundError) as exc:
        log.error("Camera unavailable: %s", exc)
        return 2

    world = World(cfg)
    db = Database(cfg)
    db.start_session(str(cfg.get("action.controller", "simulation")),
                     str(camera.source))

    try:
        controller = build_controller(cfg, world)
    except ControllerUnavailable as exc:
        log.error("Actuation layer unavailable: %s", exc)
        camera.release()
        return 3

    try:
        pipe = Pipeline(cfg, bus, world, controller, db=db)
    except ModelUnavailable as exc:
        log.error("Detection model unavailable: %s", exc)
        camera.release()
        return 4

    bus.publish(Event("SYSTEM_START", "main", {
        "controller": controller.name,
        "source": str(camera.source),
        "auto_sort": pipe.decider.auto_sort,
        "verify": pipe.decider.verify,
        "persistence": db.available,
    }))

    headless = args.headless > 0
    renderer = None
    cv2 = None
    if not headless:
        import cv2 as _cv2
        cv2 = _cv2
        from simulation.renderer import Renderer
        renderer = Renderer(cfg)
        cv2.namedWindow("Smart Municipal Waste Segregation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Municipal Waste Segregation", 1280, 900)

    log.info("Running. %s", "headless" if headless else "press q to quit")
    last_tick = time.monotonic()
    frames = 0
    exit_code = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                log.info("Camera source exhausted after %d frames", frames)
                break

            frames += 1
            detections = pipe.process_frame(frame, camera.frame_id, frame.shape[1])

            now = time.monotonic()
            world.update(now - last_tick)
            last_tick = now

            # Bin sampling, capacity alerts and the heartbeat are about
            # elapsed time, not about any one item, so they tick per frame.
            pipe.monitoring.tick(now)

            if headless:
                if frames >= args.headless:
                    break
                continue

            canvas = renderer.draw(frame, detections, pipe, world)
            cv2.imshow("Smart Municipal Waste Segregation", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 255 and not handle_key(key, controller, world, bus):
                pass

    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception:
        log.exception("Unhandled error in the main loop")
        exit_code = 1
    finally:
        camera.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        db.snapshot_bins(world.bins)
        db.end_session(frames)
        summarise(pipe, frames, db)
        if db.available:
            print_report(cfg, bus, db, routing=pipe.routing)
        db.close()
        recorder.close()

    return exit_code


def handle_key(key: int, controller, world, bus: EventBus) -> bool:
    """Operator controls. Each one goes through the controller, not around it,
    so the same keypress would drive real hardware in Phase 2."""
    if key == ord("s"):
        if world.conveyor.running:
            controller.stop_conveyor("operator pressed stop")
        else:
            controller.start_conveyor()
        return True
    if key == ord("e"):
        for bin_ in world.bins.bins.values():
            bin_.empty()
        bus.publish(Event(Topic.SYSTEM_UPDATE, "main",
                          {"reason": "all bins emptied by operator"}))
        return True
    if key == ord("f"):
        # Flip injected failures on and off live, to show the retry-then-hold
        # path working against an actuator that will not cooperate.
        rate = getattr(controller, "failure_rate", None)
        if rate is not None:
            controller.failure_rate = 0.0 if rate > 0 else 0.6
            bus.publish(Event(Topic.SYSTEM_UPDATE, "main", {
                "reason": f"injected actuator failure rate set to "
                          f"{controller.failure_rate:.0%}"},
                Severity.WARNING))
        return True
    return False


def summarise(pipe: Pipeline, frames: int, db=None) -> None:
    snap = pipe.snapshot()
    print("\n" + "=" * 62)
    print(f"  frames processed      {frames}")
    print(f"  items decided         {snap['processed']}")
    print(f"  sorted successfully   {snap['sorted_ok']}")
    print(f"  sent to manual check  {snap['manual_checks']}")
    print(f"  actuation failures    {snap['failures']}")
    print(f"  average confidence    {snap['average_confidence']:.0%}")
    if snap["by_category"]:
        print("\n  by category:")
        for name, count in sorted(snap["by_category"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<14} {count}")
    if snap["by_rule"]:
        print("\n  by safety rule:")
        for name, count in sorted(snap["by_rule"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<28} {count}")
    if db is not None and db.available:
        print(f"\n  recorded to {db.path}")
        print("  report:  streamlit run dashboard/app.py")
    elif db is not None and db.enabled:
        print("\n  persistence was unavailable this run; see the log")
    print("=" * 62)


def _wrap(text: str, width: int = 68, indent: str = "      ") -> str:
    return "\n".join(indent + line for line in textwrap.wrap(text, width))


def print_report(cfg: Config, bus: EventBus, db, routing=None) -> int:
    """Insights and recommendations, from the record only.

    Both agents are constructed here rather than being long-lived: neither
    holds state between runs, and neither can act on anything. The Planning
    Agent in particular is given no controller, so there is nothing for it to
    do beyond producing sentences addressed to a person.
    """
    from agents.analytics_agent import AnalyticsAgent
    from agents.planning_agent import PlanningAgent
    from agents.routing_agent import RoutingAgent

    if not db.available:
        print("\nNo record to report on -- persistence is unavailable.")
        return 5

    routing = routing or RoutingAgent(cfg, bus)
    insights = AnalyticsAgent(cfg, bus, db).analyse()
    recommendations = PlanningAgent(cfg, bus, routing).recommend(insights)

    print("\n" + "=" * 74)
    print("  MUNICIPAL INSIGHTS")
    print("=" * 74)
    if not insights:
        print("\n  Nothing to report.")
    for insight in insights:
        flag = "" if insight.strength == "supported" else "   [provisional]"
        print(f"\n  [{insight.kind}] {insight.headline}{flag}")
        print(_wrap(insight.detail))
        print(f"      period: {insight.period}   evidence: {insight.evidence}")

    print("\n" + "=" * 74)
    print("  RECOMMENDATIONS")
    print("=" * 74)
    print("\n  Advisory only. This system does not act on any of the following;")
    print("  each line names the person whose decision it is.")
    for rec in recommendations:
        print(f"\n  [{rec.priority}] {rec.headline}")
        print(f"      owner: {rec.owner}")
        print("      consider:")
        print(_wrap(rec.consider))
    if not recommendations:
        print("\n  Nothing to recommend.")

    if routing.routes and not routing.configured_facilities:
        print(f"\n  Note: {len(routing.routes)} downstream routes are described "
              f"generically.\n  No facility is named because none has been "
              f"verified -- see {routing.source}.")
    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
