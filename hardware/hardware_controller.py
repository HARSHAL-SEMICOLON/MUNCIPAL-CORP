"""
Phase 2 actuation: ESP32 / Arduino.

This is the other side of the seam. It implements exactly the interface
`SimulationController` implements, and the Action Agent above it does not
change by a line -- the retry-then-hold logic it relies on was written in
Stage 1 against injected simulation failures, and meets real hardware here
for the first time already tested.

Three things this file has to get right, and each of them is a way real
sorting lines go wrong:

**A reply that arrives late belongs to nothing.** If a SORT times out and the
board answers a second later, that answer must not be read as the ACK for the
*next* command -- the system would record an item as sorted into a bin the
diverter never moved to. Every reply carries the sequence number of the
command it answers, and anything else is discarded with a log line.

**An emergency stop is not a failure to retry.** `ERR ESTOP` means a person
pressed a button. Retrying is both futile and wrong, so it maps to REFUSED,
which the Action Agent already knows not to retry. A jam maps to FAILED,
which it does retry.

**A dead link stops the belt.** `check_status()` reports rather than raises,
so the Safety Agent's first rung sees an unavailable controller and stops the
conveyor instead of feeding items past a diverter that cannot move.
"""

from __future__ import annotations

import logging
import time

from core.config import Config
from core.messages import ActionResult, ActionStatus, Destination
from hardware.controller import ActionController, ControllerUnavailable
from hardware.protocol import (DeviceEvent, ErrorCode, Reply, SequenceCounter,
                               Verb, parse_line)
from hardware.transport import Transport, TransportError

log = logging.getLogger("HardwareController")

# How a device-reported error maps onto the vocabulary the agents speak.
# REFUSED is never retried; FAILED and TIMEOUT are.
ERROR_STATUS = {
    ErrorCode.ESTOP:       ActionStatus.REFUSED,
    ErrorCode.UNKNOWN_BIN: ActionStatus.REFUSED,
    ErrorCode.BAD_COMMAND: ActionStatus.REFUSED,
    ErrorCode.JAM:         ActionStatus.FAILED,
    ErrorCode.NOT_READY:   ActionStatus.FAILED,
}


class HardwareController(ActionController):
    name = "hardware"

    def __init__(self, cfg: Config, transport: Transport | None = None):
        self.cfg = cfg
        self.timeout_s = float(cfg.get("action.timeout_ms", 4000)) / 1000.0
        self.bin_angles = {
            str(k).upper(): int(v)
            for k, v in (cfg.section("hardware.bin_angles") or {}).items()
        }
        self.seq = SequenceCounter()
        self.events: list[DeviceEvent] = []
        self.estop = False
        self._link_ok = False
        # Counted, not just logged. A rising discard count is the signature of
        # a timeout set below the mechanism's travel time, and an operator
        # should be able to see it without reading a log file.
        self.discarded_replies = 0

        if transport is not None:
            self.transport: Transport | None = transport
        else:
            self.transport = self._open_serial(cfg)

        self._link_ok = self._handshake()

    # -- connection --------------------------------------------------------

    def _open_serial(self, cfg: Config) -> Transport | None:
        from hardware.transport import SerialTransport
        port = cfg.get("hardware.port")
        if not port:
            raise ControllerUnavailable(
                "action.controller is 'hardware' but hardware.port is not set "
                "in config.yaml (e.g. COM5 on Windows, /dev/ttyUSB0 on Linux)."
            )
        try:
            return SerialTransport(port, int(cfg.get("hardware.baud", 115200)))
        except TransportError as exc:
            # Constructing the controller must not explode the whole run. The
            # Safety Agent will see an unavailable controller and stop the
            # belt, which is the behaviour we want on a loose cable.
            log.error("%s", exc)
            return None

    def _handshake(self) -> bool:
        if self.transport is None or not self.transport.connected:
            return False
        reply = self._exchange(Verb.PING)
        if reply is None or not reply.ok:
            log.error("Board did not answer PING on %s; treating the actuation "
                      "layer as unavailable", self.cfg.get("hardware.port", "link"))
            return False
        log.info("Hardware link established (%s)", reply.detail or "pong")
        return True

    # -- one request, one matching reply -----------------------------------

    def _exchange(self, verb: str, *args) -> Reply | None:
        """Send a command and wait for the reply that carries its sequence.

        Returns None on timeout. Replies for other sequence numbers are
        dropped, and unsolicited events are collected rather than mistaken
        for answers.
        """
        if self.transport is None or not self.transport.connected:
            return None

        seq = self.seq.next()
        line = f"#{seq} {verb}" + ("".join(f" {a}" for a in args)) + "\n"
        try:
            self.transport.write_line(line)
        except Exception:
            log.exception("Write failed; link considered down")
            self._link_ok = False
            return None

        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("No reply to #%d %s within %.0f ms",
                            seq, verb, self.timeout_s * 1000)
                return None

            raw = self.transport.read_line(remaining)
            if raw is None:
                continue

            message = parse_line(raw)
            if message is None:
                log.debug("Ignoring unrecognised line from device: %r", raw)
                continue

            if isinstance(message, DeviceEvent):
                self._on_event(message)
                continue

            if message.seq != seq:
                # The defining bug this guards against: accepting this would
                # report an item as sorted on the strength of the previous
                # command's late answer.
                self.discarded_replies += 1
                log.warning("Discarding reply for #%d while awaiting #%d "
                            "(%d discarded so far -- if this keeps rising, "
                            "action.timeout_ms is below the mechanism's "
                            "travel time)", message.seq, seq,
                            self.discarded_replies)
                continue

            return message

    def _on_event(self, event: DeviceEvent) -> None:
        self.events.append(event)
        if event.name == "ESTOP_ENGAGED":
            self.estop = True
            log.critical("Emergency stop engaged on the device")
        elif event.name == "ESTOP_CLEARED":
            self.estop = False
            log.warning("Emergency stop cleared")
        else:
            log.info("Device event: %s %s", event.name, event.detail)

    @staticmethod
    def _result(reply: Reply | None, destination: Destination,
                started: float) -> ActionResult:
        latency = (time.perf_counter() - started) * 1000.0
        if reply is None:
            return ActionResult(ActionStatus.TIMEOUT, destination, latency,
                                detail="no reply from device")
        if reply.ok:
            return ActionResult(ActionStatus.OK, destination, latency,
                                detail=reply.detail)
        status = ERROR_STATUS.get(reply.code, ActionStatus.FAILED)
        return ActionResult(status, destination, latency,
                            detail=f"{reply.code}: {reply.detail}".strip(": "))

    # -- ActionController --------------------------------------------------

    def sort_to(self, destination: Destination, item_id: int) -> ActionResult:
        started = time.perf_counter()
        if destination.value not in self.bin_angles:
            # Caught here rather than on the board, so the message names the
            # config file the operator has to edit.
            return ActionResult(
                ActionStatus.REFUSED, destination,
                (time.perf_counter() - started) * 1000.0,
                detail=f"no servo angle configured for {destination.value}; "
                       f"add it under hardware.bin_angles in config.yaml",
            )
        reply = self._exchange(Verb.SORT, destination.value)
        return self._result(reply, destination, started)

    def start_conveyor(self) -> ActionResult:
        started = time.perf_counter()
        return self._result(self._exchange(Verb.START), Destination.NONE, started)

    def stop_conveyor(self, reason: str = "") -> ActionResult:
        started = time.perf_counter()
        # One token, so the firmware's tokeniser stays trivial.
        token = (reason or "unspecified").split()[0][:24] if reason else "unspecified"
        return self._result(self._exchange(Verb.STOP, token),
                            Destination.NONE, started)

    def hold_item(self, item_id: int, reason: str = "") -> ActionResult:
        started = time.perf_counter()
        return self._result(self._exchange(Verb.HOLD, item_id),
                            Destination.NONE, started)

    def check_status(self) -> dict:
        """Never raises. An agent asking 'are you there?' gets an answer."""
        if self.transport is None or not self.transport.connected:
            return {"available": False, "controller": self.name,
                    "detail": "no link to the device"}
        reply = self._exchange(Verb.STATUS)
        if reply is None or not reply.ok:
            self._link_ok = False
            return {"available": False, "controller": self.name,
                    "detail": "device did not answer STATUS"}
        self._link_ok = True
        return {
            "available": not self.estop,
            "controller": self.name,
            "estop": self.estop,
            "detail": reply.detail,
            "events": len(self.events),
            "discarded_replies": self.discarded_replies,
        }

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
