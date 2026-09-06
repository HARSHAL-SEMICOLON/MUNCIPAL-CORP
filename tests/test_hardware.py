"""
Phase 2 tests -- run with no hardware attached.

The HardwareController talks to a simulated ESP32 that speaks the real wire
protocol, so every path below is the same code that will drive a servo:
sequence matching, timeouts, jams, emergency stops, and a dead link.

The most valuable test here is `test_late_reply_is_not_matched_to_the_next_command`.
That bug does not announce itself -- the system keeps running and quietly
records items as sorted into bins the diverter never reached. It is much
easier to design out now than to find on a bench at 1am.

Run:  python -m tests.test_hardware
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from core.config import Config
from core.event_bus import EventBus
from core.messages import ActionStatus, Destination, SafetyVerdict, Action, Severity
from agents.action_agent import ActionAgent
from hardware.controller import ActionController
from hardware.fake_device import FakeDevice
from hardware.hardware_controller import HardwareController
from hardware.protocol import (Command, DeviceEvent, Reply, SequenceCounter,
                               parse_command, parse_line)
from hardware.transport import LoopbackTransport
from tests.test_pipeline import PASSED, FAILED, check

logging.disable(logging.CRITICAL)

ANGLES = {"RECYCLING": 30, "ORGANIC": 60, "E_WASTE": 90,
          "HAZARDOUS": 120, "REJECT": 150, "MANUAL_CHECK": 180}


def rig(travel_ms: float = 20.0, timeout_ms: int = 500, **device_kwargs):
    cfg = Config.load()
    cfg._data.setdefault("action", {})["timeout_ms"] = timeout_ms
    cfg._data.setdefault("hardware", {})["bin_angles"] = dict(ANGLES)
    device = FakeDevice(bin_angles=ANGLES, travel_ms=travel_ms, **device_kwargs)
    controller = HardwareController(cfg, transport=LoopbackTransport(device))
    return cfg, device, controller


# ---------------------------------------------------------------------------
#  The protocol itself
# ---------------------------------------------------------------------------

def test_protocol_round_trips():
    line = Command(seq=7, verb="SORT", args=("ORGANIC",)).encode()
    check("a command encodes to one readable line",
          line == "#7 SORT ORGANIC\n", repr(line))
    back = parse_command(line)
    check("and parses back on the device side",
          back is not None and back.seq == 7 and back.args == ("ORGANIC",))

    ok = parse_line("#7 OK angle=60")
    check("an OK reply carries its sequence",
          isinstance(ok, Reply) and ok.ok and ok.seq == 7)

    err = parse_line("#8 ERR JAM diverter did not reach 60deg")
    check("an ERR reply carries a code and a detail",
          isinstance(err, Reply) and not err.ok and err.code == "JAM"
          and "60deg" in err.detail, f"{err}")

    event = parse_line("!EVT ESTOP_ENGAGED operator")
    check("an unsolicited event is not mistaken for a reply",
          isinstance(event, DeviceEvent) and event.name == "ESTOP_ENGAGED")


def test_noise_is_ignored_not_fatal():
    for junk in ("", "   ", "ets Jun  8 2016 00:22:57", "\x00\xff garbage",
                 "rst:0x1 (POWERON_RESET)"):
        check(f"line noise {junk[:18]!r} is ignored",
              parse_line(junk) is None)


def test_sequence_counter_wraps():
    counter = SequenceCounter(value=9998, limit=9999)
    check("sequence increments", counter.next() == 9999)
    check("and wraps rather than growing without bound", counter.next() == 1)


# ---------------------------------------------------------------------------
#  The controller against a simulated board
# ---------------------------------------------------------------------------

def test_sorts_against_a_simulated_board():
    _, device, controller = rig()
    result = controller.sort_to(Destination.ORGANIC, item_id=1)
    check("a sort succeeds", result.ok, f"{result.status.value} {result.detail}")
    check("the servo was commanded to the configured angle",
          device.last_angle == 60, f"angle={device.last_angle}")
    check("and the latency is measured, not assumed",
          result.latency_ms > 0, f"{result.latency_ms:.1f} ms")


def test_implements_the_same_interface_as_the_simulation():
    _, _, controller = rig()
    check("HardwareController is an ActionController",
          isinstance(controller, ActionController))
    check("with nothing abstract left over",
          not getattr(type(controller), "__abstractmethods__", set()))


def test_unconfigured_bin_is_refused_not_guessed():
    cfg, device, controller = rig()
    controller.bin_angles.pop("HAZARDOUS")
    result = controller.sort_to(Destination.HAZARDOUS, item_id=2)
    check("an unmapped destination is REFUSED",
          result.status is ActionStatus.REFUSED, result.status.value)
    check("and names the config key to fix",
          "bin_angles" in result.detail, result.detail)
    check("the servo was never commanded", device.last_angle is None)


def test_a_jam_is_retryable_but_an_estop_is_not():
    """The distinction the Action Agent depends on."""
    _, device, controller = rig()
    device.jam_rate = 1.0
    jam = controller.sort_to(Destination.ORGANIC, item_id=3)
    check("a jam reports FAILED, which the Action Agent retries",
          jam.status is ActionStatus.FAILED, jam.status.value)

    device.jam_rate = 0.0
    device.engage_estop()
    stopped = controller.sort_to(Destination.ORGANIC, item_id=4)
    check("an emergency stop reports REFUSED, which it does not retry",
          stopped.status is ActionStatus.REFUSED, stopped.status.value)
    check("and the controller noticed the e-stop event", controller.estop)


def test_a_silent_board_times_out():
    _, device, controller = rig()
    device.silent = True
    result = controller.sort_to(Destination.ORGANIC, item_id=5)
    check("a wedged board produces TIMEOUT, not a hang",
          result.status is ActionStatus.TIMEOUT, result.status.value)


def test_a_slow_mechanism_times_out():
    """A timeout shorter than the servo's travel fails every command --
    a real bring-up mistake, and it should be obvious rather than mysterious.

    The link is brought up healthy first and the mechanism slowed afterwards,
    so this tests the command path rather than a failed handshake.
    """
    _, device, controller = rig(travel_ms=20.0, timeout_ms=200)
    check("the link came up before the mechanism slowed", controller.available)

    device.travel_ms = 900.0
    result = controller.sort_to(Destination.RECYCLING, item_id=6)
    check("a mechanism slower than the timeout reports TIMEOUT",
          result.status is ActionStatus.TIMEOUT, result.status.value)


def test_late_reply_is_not_matched_to_the_next_command():
    """The bug worth designing out.

    A SORT times out; its answer arrives during the *next* command. Accepting
    it would record an item as sorted into a bin the diverter never reached,
    and nothing would look wrong.
    """
    # Healthy link first, then a mechanism that slows down -- which is how
    # this actually happens: a belt that was fine all morning starts binding.
    _, device, controller = rig(travel_ms=20.0, timeout_ms=200)
    device.travel_ms = 900.0

    first = controller.sort_to(Destination.RECYCLING, item_id=7)
    check("the first command times out",
          first.status is ActionStatus.TIMEOUT, first.status.value)

    # The board is quick again, but the stale reply is still in the pipe.
    device.travel_ms = 5.0
    second = controller.sort_to(Destination.ORGANIC, item_id=8)

    check("the stale reply was actually seen and discarded",
          controller.discarded_replies == 1,
          f"discarded={controller.discarded_replies}")
    check("and the second command got its own ACK, not the stale one",
          second.ok and device.last_angle == 60,
          f"{second.status.value} angle={device.last_angle}")
    sorts = [c for c in device.commands if c.verb == "SORT"]
    check("both commands carried distinct sequence numbers",
          len({c.seq for c in sorts}) == len(sorts),
          f"seqs={[c.seq for c in sorts]}")


def test_status_reports_rather_than_raises():
    _, device, controller = rig()
    check("a healthy link reports available", controller.available)

    device.engage_estop()
    controller.check_status()
    status = controller.check_status()
    check("an engaged e-stop makes the controller unavailable",
          not status["available"], str(status))

    controller.transport.close()
    status = controller.check_status()
    check("a closed link reports rather than raising",
          status["available"] is False and "link" in status["detail"],
          str(status))


def test_dead_link_is_reported_not_raised():
    cfg = Config.load()
    cfg._data.setdefault("action", {})["timeout_ms"] = 200
    device = FakeDevice(bin_angles=ANGLES, travel_ms=10.0)
    device.silent = True
    controller = HardwareController(cfg, transport=LoopbackTransport(device))
    check("a board that never answers PING is unavailable at construction",
          not controller.available)
    check("but constructing it did not raise", True)


# ---------------------------------------------------------------------------
#  The seam holds: the Action Agent is unchanged
# ---------------------------------------------------------------------------

def test_action_agent_drives_hardware_unchanged():
    """The whole return on building the seam in Stage 1.

    The retry-then-hold logic was written against injected *simulation*
    failures. Here it meets a jamming board for the first time, with no edit.
    """
    cfg, device, controller = rig()
    device.jam_rate = 1.0
    bus = EventBus()
    agent = ActionAgent(cfg, bus, controller)

    verdict = SafetyVerdict(True, Action.SORT, Destination.ORGANIC,
                            "high-confidence organic", rule="R7_APPROVED")
    result = agent.execute(verdict, item_id=9, label="banana")

    attempts = len([c for c in device.commands if c.verb == "SORT"])
    expected = int(cfg.get("action.max_retries", 2)) + 1
    check("a jamming diverter is retried to the configured limit",
          attempts == expected, f"attempts={attempts} expected={expected}")
    check("and the item is then held, not recorded as sorted",
          not result.ok and any(c.verb == "HOLD" for c in device.commands),
          f"status={result.status.value}")


def test_config_selects_hardware():
    from hardware.controller import build_controller
    cfg = Config.load()
    cfg._data["action"]["controller"] = "hardware"
    cfg._data.setdefault("hardware", {})["port"] = None
    try:
        build_controller(cfg)
        check("hardware without a port is refused with a clear message", False)
    except Exception as exc:
        check("hardware without a port names the missing config key",
              "hardware.port" in str(exc), str(exc)[:70])
    finally:
        cfg._data["action"]["controller"] = "simulation"


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
