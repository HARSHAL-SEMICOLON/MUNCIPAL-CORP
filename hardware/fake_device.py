"""
A simulated ESP32 that speaks the real wire protocol.

This is the piece that makes Phase 2 testable before any hardware is bought.
The HardwareController talks to it exactly as it would talk to a real board --
same lines, same sequence numbers, same timeouts -- so the whole actuation
path can be exercised on a laptop with nothing plugged in.

It is deliberately unhelpful when asked to be. Real mechanisms jam, servos
take time to travel, emergency stops get pressed mid-command, and replies
arrive after the host gave up waiting. Every one of those is reproducible
here on demand, which is the only way the controller's failure handling gets
written before it is needed rather than after something breaks.

This is not firmware. `firmware/esp32_sorter/` holds the sketch that runs on
the real board; this file is its stand-in, and the protocol module is the
contract they both implement.
"""

from __future__ import annotations

import random

from hardware.protocol import (Command, ErrorCode, EventName, Verb,
                               parse_command)


class FakeDevice:
    """An in-process sorting rig.

    `travel_ms` models how long a servo takes to reach its angle -- the
    controller's timeout has to be larger than this or every command times
    out, which is a genuine bring-up mistake worth being able to reproduce.
    """

    def __init__(self, bin_angles: dict[str, int] | None = None,
                 travel_ms: float = 350.0, seed: int = 20260905):
        self.bin_angles = {k.upper(): v for k, v in (bin_angles or {}).items()}
        self.travel_ms = travel_ms
        self.rng = random.Random(seed)

        self.conveyor_running = False
        self.estop = False
        self.last_angle: int | None = None
        self.commands: list[Command] = []
        self.pending_events: list[str] = []

        # Failure injection -- all off by default.
        self.jam_rate = 0.0
        self.silent = False          # accept commands, answer nothing (dead link)
        self.reply_delay_ms = 0.0    # extra latency on top of travel

    # -- things a test or an operator can do to it -------------------------

    def engage_estop(self) -> None:
        self.estop = True
        self.conveyor_running = False
        self.pending_events.append(f"!EVT {EventName.ESTOP_ENGAGED} operator\n")

    def clear_estop(self) -> None:
        self.estop = False
        self.pending_events.append(f"!EVT {EventName.ESTOP_CLEARED} operator\n")

    def report_bin_level(self, bin_name: str, percent: int) -> None:
        self.pending_events.append(
            f"!EVT {EventName.BIN_LEVEL} {bin_name} {percent}\n")

    # -- the protocol ------------------------------------------------------

    def handle(self, line: str) -> list[str]:
        """One command in, zero or more lines out."""
        out = self.pending_events[:]
        self.pending_events.clear()

        command = parse_command(line)
        if command is None:
            return out

        self.commands.append(command)

        if self.silent:
            # A live cable and a wedged firmware look identical from the host.
            return out

        reply = self._respond(command)
        if reply:
            out.append(reply)
        return out

    def _respond(self, command: Command) -> str:
        seq, verb = command.seq, command.verb

        # The emergency stop outranks every command except the ones that ask
        # about it. A real e-stop is wired to cut motor power in hardware as
        # well; refusing in firmware is the second line of defence, not the
        # first.
        if self.estop and verb not in (Verb.PING, Verb.STATUS, Verb.STOP):
            return f"#{seq} ERR {ErrorCode.ESTOP} emergency stop engaged\n"

        if verb == Verb.PING:
            return f"#{seq} OK pong\n"

        if verb == Verb.STATUS:
            return (f"#{seq} OK running={int(self.conveyor_running)} "
                    f"estop={int(self.estop)} angle={self.last_angle}\n")

        if verb == Verb.START:
            self.conveyor_running = True
            return f"#{seq} OK motor on\n"

        if verb == Verb.STOP:
            self.conveyor_running = False
            return f"#{seq} OK motor off\n"

        if verb == Verb.HOLD:
            if not command.args:
                return f"#{seq} ERR {ErrorCode.BAD_COMMAND} HOLD needs an item id\n"
            self.conveyor_running = False
            return f"#{seq} OK holding {command.args[0]}\n"

        if verb == Verb.SORT:
            if not command.args:
                return f"#{seq} ERR {ErrorCode.BAD_COMMAND} SORT needs a bin\n"
            bin_name = command.args[0].upper()
            angle = self.bin_angles.get(bin_name)
            if angle is None:
                return (f"#{seq} ERR {ErrorCode.UNKNOWN_BIN} "
                        f"no angle configured for {bin_name}\n")
            if self.jam_rate and self.rng.random() < self.jam_rate:
                # The servo was commanded but its feedback never reached the
                # target. The item is somewhere unknown; the host must not
                # record it as sorted.
                return (f"#{seq} ERR {ErrorCode.JAM} "
                        f"diverter did not reach {angle}deg\n")
            self.last_angle = angle
            return f"#{seq} OK angle={angle}\n"

        return f"#{seq} ERR {ErrorCode.BAD_COMMAND} unknown verb {verb}\n"

    @property
    def latency_ms(self) -> float:
        return self.travel_ms + self.reply_delay_ms
