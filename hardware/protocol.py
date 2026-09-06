"""
The wire protocol between this system and the sorting hardware.

Line-based ASCII, one message per line, `\\n` terminated. That is a deliberate
choice over a packed binary format: a student debugging a servo at 1am can
open a serial monitor and read exactly what is being said, and can type a
command by hand to test a mechanism without running the whole system. The
bandwidth cost is irrelevant at a few commands per second.

    host -> device    #<seq> <VERB> [args...]
    device -> host    #<seq> OK [detail]
                      #<seq> ERR <code> [detail]
                      !EVT <NAME> [detail]          (unsolicited)

**Sequence numbers are the part that matters.** Without them, a reply that
arrives after its command timed out gets matched to the *next* command, and
the system believes a servo moved when it did not. The controller discards
any reply whose sequence number is not the one it is waiting for, and
`test_late_reply_is_not_matched_to_the_next_command` holds it to that.

Unsolicited events carry `!` rather than a sequence number, because the
device raises them on its own schedule -- an emergency stop is not an answer
to anything the host asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class Verb:
    PING   = "PING"      # is the link alive
    STATUS = "STATUS"    # report link, e-stop, last servo position
    SORT   = "SORT"      # SORT <BIN>  -- move the diverter
    START  = "START"     # conveyor motor on
    STOP   = "STOP"      # STOP <reason> -- conveyor motor off
    HOLD   = "HOLD"      # HOLD <item_id> -- park the item


class ErrorCode:
    ESTOP        = "ESTOP"         # emergency stop is engaged
    UNKNOWN_BIN  = "UNKNOWN_BIN"   # no servo angle configured for that bin
    JAM          = "JAM"           # mechanism did not reach position
    NOT_READY    = "NOT_READY"     # still moving from the last command
    BAD_COMMAND  = "BAD_COMMAND"


class EventName:
    ESTOP_ENGAGED  = "ESTOP_ENGAGED"
    ESTOP_CLEARED  = "ESTOP_CLEARED"
    BIN_LEVEL      = "BIN_LEVEL"      # BIN_LEVEL <bin> <percent>
    BOOT           = "BOOT"


@dataclass(frozen=True)
class Command:
    seq: int
    verb: str
    args: tuple[str, ...] = ()

    def encode(self) -> str:
        parts = [f"#{self.seq}", self.verb, *self.args]
        return " ".join(str(p) for p in parts) + "\n"


@dataclass(frozen=True)
class Reply:
    seq: int
    ok: bool
    code: str = ""
    detail: str = ""


@dataclass(frozen=True)
class DeviceEvent:
    name: str
    detail: str = ""


_REPLY = re.compile(r"^#(\d+)\s+(OK|ERR)\b\s*(.*)$")
_EVENT = re.compile(r"^!EVT\s+(\S+)\s*(.*)$")
_COMMAND = re.compile(r"^#(\d+)\s+([A-Z_]+)\s*(.*)$")


def parse_line(line: str) -> Reply | DeviceEvent | None:
    """Turn one line from the device into a Reply or an event.

    Returns None for anything unrecognised -- boot banners, debug prints from
    the firmware, line noise on a long cable. Those are logged and ignored
    rather than raising: a stray character on a serial line must not stop a
    conveyor.
    """
    line = (line or "").strip()
    if not line:
        return None

    event = _EVENT.match(line)
    if event:
        return DeviceEvent(name=event.group(1), detail=event.group(2).strip())

    reply = _REPLY.match(line)
    if reply:
        seq, status, rest = int(reply.group(1)), reply.group(2), reply.group(3).strip()
        if status == "OK":
            return Reply(seq=seq, ok=True, detail=rest)
        # ERR lines carry a code first, then optional detail.
        bits = rest.split(None, 1)
        return Reply(seq=seq, ok=False,
                     code=bits[0] if bits else ErrorCode.BAD_COMMAND,
                     detail=bits[1].strip() if len(bits) > 1 else "")
    return None


def parse_command(line: str) -> Command | None:
    """The device side of the protocol. Used by the simulated device, and a
    reference for whoever writes the firmware."""
    line = (line or "").strip()
    match = _COMMAND.match(line)
    if not match:
        return None
    args = tuple(a for a in match.group(3).split() if a)
    return Command(seq=int(match.group(1)), verb=match.group(2), args=args)


@dataclass
class SequenceCounter:
    """Monotonic, wrapping well short of anything a firmware int would overflow."""
    value: int = 0
    limit: int = 9999

    def next(self) -> int:
        self.value = 1 if self.value >= self.limit else self.value + 1
        return self.value
