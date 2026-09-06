# Phase 2 — the physical line

The AI layer does not change. `HardwareController` replaces
`SimulationController` below the seam, and the ten agents above it are the
same code that has been running in simulation all along.

```yaml
action:
  controller: hardware       # ← was "simulation"

hardware:
  port: "COM5"               # /dev/ttyUSB0 on Linux
  baud: 115200
  bin_angles:
    RECYCLING:     30
    ORGANIC:       60
    E_WASTE:       90
    HAZARDOUS:    120
    REJECT:       150
    MANUAL_CHECK: 180
```

That is the migration.

---

## Test it before you buy anything

```bash
python -m tests.test_hardware
```

38 checks, no board attached. `hardware/fake_device.py` is a simulated ESP32
that speaks the real wire protocol, so the controller exercises the same code
paths a real board would drive: sequence matching, timeouts, jams, emergency
stops, a dead link, and a reply that arrives after its command gave up.

This is the same principle as `action.simulated_failure_rate` in Phase 1 — the
failure handling gets written and tested *before* it is needed, not after
something breaks on a bench at 1am.

---

## The protocol

Line-based ASCII, one message per line. Chosen over a packed binary format so
you can open a serial monitor, read exactly what is being said, and type a
command by hand to test a mechanism without running the whole system.

```
host -> board    #<seq> <VERB> [args]
board -> host    #<seq> OK [detail]
                 #<seq> ERR <code> [detail]
                 !EVT <NAME> [detail]        (unsolicited)
```

| Verb | Meaning |
|------|---------|
| `PING` | is the link alive |
| `STATUS` | running / e-stop / last servo angle |
| `SORT <BIN>` | move the diverter, answer when it has **arrived** |
| `START` | conveyor motor on |
| `STOP <reason>` | conveyor motor off |
| `HOLD <item>` | park the item |

| Error code | Host treats it as | Retried? |
|-----------|-------------------|----------|
| `ESTOP` | REFUSED | no — a person pressed a button |
| `UNKNOWN_BIN` | REFUSED | no — config is wrong, retrying will not fix it |
| `BAD_COMMAND` | REFUSED | no |
| `JAM` | FAILED | yes |
| `NOT_READY` | FAILED | yes |
| *(no reply)* | TIMEOUT | yes |

Try it by hand at 115200 baud:

```
#1 PING
#2 SORT ORGANIC
#3 STATUS
```

---

## Three things the firmware must get right

**Always echo the sequence number you were given.** The host discards any
reply whose sequence does not match the command it is waiting for. This is
not pedantry: without it, a SORT that timed out and answered a second late
would be read as the ACK for the *next* command, and the system would record
an item as sorted into a bin the diverter never reached. Nothing would look
wrong. `test_late_reply_is_not_matched_to_the_next_command` exists for this.

**Only answer `OK` once the servo has arrived**, not when the command was
accepted. Answering early means the host believes an item was diverted while
the arm is still travelling, and the next item goes to the wrong place.

**Wire the emergency stop to cut motor power in hardware.** The firmware
check is a second line of defence. Never rely on software alone to stop a
machine that can injure someone. Note also that clearing the e-stop does
*not* restart the belt — in the firmware or in the Safety Agent. A machine
must not start itself because someone released a button.

---

## Bring-up order

Do these in order; each one isolates a different thing that can be wrong.

1. **Flash and open a serial monitor.** You should see
   `!EVT BOOT esp32_sorter ready`. Type `#1 PING`, expect `#1 OK pong`.
2. **Servo, unloaded, unmounted.** `#2 SORT ORGANIC` → `#2 OK angle=60`. Check
   the arm actually reaches 60°, and measure how long it takes.
3. **Set `SERVO_SETTLE_MS`** to that measured time plus margin, and
   `action.timeout_ms` in `config.yaml` to roughly three times it. A timeout
   below the travel time makes every command fail —
   `test_a_slow_mechanism_times_out` reproduces exactly that.
4. **Motor, belt empty.** `#3 START`, `#4 STOP manual`.
5. **Emergency stop.** Press it mid-run; expect `!EVT ESTOP_ENGAGED` and the
   belt to stop. Then `#5 SORT ORGANIC` → `#5 ERR ESTOP`.
6. **Only now** point `action.controller` at `hardware` and run `main.py`.

If step 6 misbehaves, check `discarded_replies` in `check_status()` first. A
rising count means replies are arriving after their timeout — go back to
step 3.

---

## Two sources of truth to keep in step

`BIN_ANGLES` in the sketch and `hardware.bin_angles` in `config.yaml` describe
the same physical angles in two places. If they drift, items go to the wrong
bin and **nothing reports an error** — the host asks for `ORGANIC`, the board
moves to whatever it thinks `ORGANIC` means, and both are satisfied.

The host refuses to command a destination it has no angle for, which catches
a missing entry. It cannot catch a *disagreeing* entry. Change them together,
and after re-aiming a bracket, re-check both.

---

## Parts

Nothing here is a specific product recommendation — these are categories, and
you should choose against what your supplier actually stocks.

- ESP32 dev board
- Servo sized for the diverter's load and travel; a hobby servo will not move
  a heavy flap reliably
- Motor driver rated above the conveyor motor's stall current, not its
  running current
- Separate supply for motors, common ground with the ESP32
- Latching emergency stop switch, wired to cut motor power directly
- Ultrasonic distance sensors for bin level (optional; the system works
  without them, using its own counts)
