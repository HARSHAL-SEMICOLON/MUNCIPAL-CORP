"""
How bytes reach the board.

Two implementations, one interface. `SerialTransport` is a USB cable to an
ESP32; `LoopbackTransport` is the simulated device in this process. The
HardwareController cannot tell them apart, which is what lets the whole
actuation path be tested with nothing plugged in.

pyserial is imported lazily. This module has to be importable on a machine
that has never had it installed -- the tests do exactly that -- and a missing
driver should surface as a clear message at connect time rather than an
ImportError at start-up.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque

log = logging.getLogger("transport")


class TransportError(RuntimeError):
    pass


class Transport(ABC):
    @abstractmethod
    def write_line(self, line: str) -> None: ...

    @abstractmethod
    def read_line(self, timeout_s: float) -> str | None:
        """One line, or None if none arrived inside the timeout."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...


class SerialTransport(Transport):
    """A USB or UART link to the board."""

    def __init__(self, port: str, baud: int = 115200, connect_timeout_s: float = 2.0):
        try:
            import serial
        except ImportError as exc:
            raise TransportError(
                "pyserial is not installed. pip install pyserial"
            ) from exc

        try:
            self._serial = serial.Serial(port=port, baudrate=baud,
                                         timeout=0.05, write_timeout=1.0)
        except Exception as exc:
            raise TransportError(
                f"Could not open {port} at {baud} baud: {exc}. Check the cable, "
                f"that the board is powered, and that no serial monitor is "
                f"holding the port."
            ) from exc

        # An ESP32 resets when the port opens and prints a boot banner. Give
        # it a moment, then discard whatever it said; the protocol parser
        # ignores unrecognised lines anyway, but starting clean avoids a pile
        # of confusing log warnings on every connect.
        time.sleep(connect_timeout_s)
        try:
            self._serial.reset_input_buffer()
        except Exception:
            pass
        self._buffer = ""
        log.info("Serial link open on %s at %d baud", port, baud)

    def write_line(self, line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        self._serial.write(line.encode("ascii", errors="replace"))

    def read_line(self, timeout_s: float) -> str | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                return line
            chunk = self._serial.read(256)
            if chunk:
                self._buffer += chunk.decode("ascii", errors="replace")
        if "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            return line
        return None

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        try:
            return bool(self._serial.is_open)
        except Exception:
            return False


class LoopbackTransport(Transport):
    """The simulated device, wired straight into this process.

    Time is real here: the device's stated travel time is actually slept, so
    a timeout that is too short for the mechanism fails in a test the same
    way it would fail on a bench.
    """

    def __init__(self, device):
        self.device = device
        self._inbox: deque[str] = deque()
        self._open = True

    def write_line(self, line: str) -> None:
        if not self._open:
            raise TransportError("loopback transport is closed")
        for reply in self.device.handle(line):
            self._inbox.append(reply.rstrip("\n"))

    def read_line(self, timeout_s: float) -> str | None:
        if not self._inbox:
            return None
        wait = min(timeout_s, self.device.latency_ms / 1000.0)
        if wait > 0:
            time.sleep(wait)
        if self.device.latency_ms / 1000.0 > timeout_s:
            # The mechanism is slower than the host is willing to wait. The
            # reply is left in the inbox on purpose -- that is exactly how a
            # late reply arrives during the *next* command on real hardware,
            # and the controller has to survive it.
            return None
        return self._inbox.popleft()

    def close(self) -> None:
        self._open = False

    @property
    def connected(self) -> bool:
        return self._open
