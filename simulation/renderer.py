"""
The control-room view: camera, decision readout, conveyor, bins, alerts.

Drawn with OpenCV into a single window rather than a web dashboard, for one
practical reason: the sorting loop runs at camera frame rate, and Streamlit
re-runs its script top to bottom on every interaction. Fighting that for a
live belt is a demo that fails in the room. Stage 3 adds a Streamlit view for
the parts that are genuinely page-shaped -- history, charts, daily reports --
reading from SQLite while this window keeps the real-time job.

The palette is deliberately municipal: slate, teal and amber, no neon.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.config import Config
from core.messages import Category, Destination, Severity
from simulation.waste_objects import ItemState

W, H = 1280, 900
CAM_X0, CAM_Y0, CAM_W = 16, 62, 840
PANEL_X0 = CAM_W + 32
BELT_Y0, BELT_Y1 = 556, 690
BIN_Y0 = 706
FOOT_Y0 = 838

# BGR
BG        = (26, 24, 22)
PANEL     = (42, 39, 36)
PANEL_2   = (54, 50, 46)
LINE      = (74, 70, 66)
TEXT      = (234, 233, 230)
MUTED     = (152, 150, 146)
DIM       = (110, 108, 104)
TEAL      = (176, 152, 46)
AMBER     = (48, 168, 232)
RED       = (60, 62, 214)
GREEN     = (104, 176, 92)
INDIGO    = (188, 116, 96)
GREY      = (120, 118, 116)

CATEGORY_COLOUR = {
    Category.RECYCLABLE:   TEAL,
    Category.GLASS:        TEAL,
    Category.METAL:        TEAL,
    Category.PAPER:        TEAL,
    Category.ORGANIC:      GREEN,
    Category.E_WASTE:      INDIGO,
    Category.HAZARDOUS:    RED,
    Category.REJECT:       GREY,
    Category.MANUAL_CHECK: AMBER,
}

DESTINATION_COLOUR = {
    Destination.RECYCLING:    TEAL,
    Destination.ORGANIC:      GREEN,
    Destination.E_WASTE:      INDIGO,
    Destination.HAZARDOUS:    RED,
    Destination.REJECT:       GREY,
    Destination.MANUAL_CHECK: AMBER,
}

SEVERITY_COLOUR = {
    Severity.INFO: MUTED,
    Severity.WARNING: AMBER,
    Severity.ALERT: RED,
    Severity.CRITICAL: RED,
}

F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, x, y, scale=0.44, colour=TEXT, thick=1, font=F):
    cv2.putText(img, s, (int(x), int(y)), font, scale, colour, thick, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, fill=PANEL, border=LINE):
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), fill, -1)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), border, 1)


def clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 1] + "…"


def wrap(s: str, width: int, lines: int) -> list[str]:
    out, current = [], ""
    for word in s.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            out.append(current)
            current = word
            if len(out) == lines:
                break
    if current and len(out) < lines:
        out.append(current)
    return out[:lines]


class Renderer:
    def __init__(self, cfg: Config):
        self.line_position = float(cfg.get("decision_line.position", 0.55))
        self.title = str(cfg.get("system.name", "Smart Waste System"))
        self.controller_name = str(cfg.get("action.controller", "simulation"))

    # -- public ------------------------------------------------------------

    def draw(self, frame, detections, pipe, world) -> np.ndarray:
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        self._header(canvas, world)
        cam_h = self._camera(canvas, frame, detections, pipe)
        self._readout(canvas, pipe, cam_h)
        self._conveyor(canvas, world)
        self._bins(canvas, world)
        self._footer(canvas, pipe)
        return canvas

    # -- sections ----------------------------------------------------------

    def _header(self, img, world):
        panel(img, 0, 0, W, 46, PANEL, PANEL)
        text(img, self.title.upper(), 18, 30, 0.6, TEXT, 1, FD)

        # Right-aligned by measurement, so a longer controller name cannot
        # run off the edge of the header.
        mode = f"PHASE 1  |  {self.controller_name.upper()}"
        mode_w = cv2.getTextSize(mode, F, 0.4, 1)[0][0]
        text(img, mode, W - 18 - mode_w, 28, 0.4, DIM)

        running = world.conveyor.running
        state = "BELT RUNNING" if running else "BELT STOPPED"
        colour = GREEN if running else RED
        state_w = cv2.getTextSize(state, F, 0.46, 1)[0][0]
        state_x = W - 18 - mode_w - 24 - state_w
        cv2.circle(img, (state_x - 14, 23), 5, colour, -1)
        text(img, state, state_x, 28, 0.46, colour)

    def _camera(self, img, frame, detections, pipe) -> int:
        if frame is None:
            panel(img, CAM_X0, CAM_Y0, CAM_X0 + CAM_W, CAM_Y0 + 472)
            text(img, "NO CAMERA SIGNAL", CAM_X0 + 300, CAM_Y0 + 240, 0.6, RED, 1, FD)
            return 472

        scale = CAM_W / frame.shape[1]
        view = cv2.resize(frame, (CAM_W, int(frame.shape[0] * scale)))
        cam_h = view.shape[0]

        # The decision line, drawn where the agents actually commit.
        line_x = int(CAM_W * self.line_position)
        overlay = view.copy()
        cv2.line(overlay, (line_x, 0), (line_x, cam_h), AMBER, 1)
        cv2.addWeighted(overlay, 0.7, view, 0.3, 0, view)
        text(view, "DECISION LINE", line_x + 6, 18, 0.36, AMBER)

        for det in detections:
            x1, y1, x2, y2 = (int(v * scale) for v in det.bbox)
            done = pipe.committed(det.track_id)
            colour = TEAL if done else MUTED
            cv2.rectangle(view, (x1, y1), (x2, y2), colour, 2 if done else 1)
            tag = f"{det.label} {det.confidence:.0%}"
            if done:
                tag += "  COMMITTED"
            tw = cv2.getTextSize(tag, F, 0.4, 1)[0][0]
            cv2.rectangle(view, (x1, y1 - 17), (x1 + tw + 10, y1), colour, -1)
            text(view, tag, x1 + 5, y1 - 5, 0.4, (20, 20, 20))

        img[CAM_Y0:CAM_Y0 + cam_h, CAM_X0:CAM_X0 + CAM_W] = view
        cv2.rectangle(img, (CAM_X0, CAM_Y0), (CAM_X0 + CAM_W, CAM_Y0 + cam_h), LINE, 1)
        text(img, "LIVE CAMERA", CAM_X0, CAM_Y0 - 8, 0.42, MUTED)
        return cam_h

    def _readout(self, img, pipe, cam_h):
        x0, x1 = PANEL_X0, W - 16
        y0, y1 = CAM_Y0, CAM_Y0 + cam_h
        panel(img, x0, y0, x1, y1)
        text(img, "AI DECISION", x0, y0 - 8, 0.42, MUTED)

        rec = pipe.last_record
        if rec is None:
            text(img, "Waiting for an object to", x0 + 16, y0 + 40, 0.44, DIM)
            text(img, "cross the decision line.", x0 + 16, y0 + 62, 0.44, DIM)
            return

        colour = CATEGORY_COLOUR.get(rec.category, MUTED)
        y = y0 + 34

        text(img, "OBJECT", x0 + 16, y, 0.36, DIM)
        text(img, clip(rec.label.upper(), 24), x0 + 16, y + 22, 0.56, TEXT, 1, FD)
        y += 52

        for label, value in (("MATERIAL", rec.material.value),
                             ("CATEGORY", rec.category.value)):
            text(img, label, x0 + 16, y, 0.36, DIM)
            shade = colour if label == "CATEGORY" else TEXT
            text(img, clip(value, 22), x0 + 16, y + 20, 0.46, shade)
            y += 46

        # Confidence, with the band it fell into named rather than implied.
        text(img, "CONFIDENCE", x0 + 16, y, 0.36, DIM)
        conf = rec.detection_confidence
        text(img, f"{conf:.0%}", x0 + 16, y + 22, 0.6, TEXT, 1, FD)
        bar_x0, bar_x1 = x0 + 100, x1 - 16
        cv2.rectangle(img, (bar_x0, y + 8), (bar_x1, y + 20), PANEL_2, -1)
        fill = int(bar_x0 + (bar_x1 - bar_x0) * min(1.0, conf))
        band = GREEN if conf >= pipe.decider.auto_sort else (
            AMBER if conf >= pipe.decider.verify else RED)
        cv2.rectangle(img, (bar_x0, y + 8), (fill, y + 20), band, -1)
        y += 50

        cv2.line(img, (x0 + 16, y - 8), (x1 - 16, y - 8), LINE, 1)

        text(img, "ACTION", x0 + 16, y + 14, 0.36, DIM)
        dest_colour = DESTINATION_COLOUR.get(rec.destination, MUTED)
        text(img, rec.action.value, x0 + 16, y + 38, 0.56, dest_colour, 1, FD)
        text(img, f"-> {rec.destination.value}", x0 + 16, y + 60, 0.44, dest_colour)
        y += 82

        text(img, f"SAFETY RULE  {rec.safety_rule}", x0 + 16, y, 0.36, DIM)
        y += 18
        for line in wrap(rec.reason, 34, 4):
            text(img, line, x0 + 16, y, 0.38, MUTED)
            y += 16

    def _conveyor(self, img, world):
        panel(img, 16, BELT_Y0, W - 16, BELT_Y1)
        text(img, "SIMULATED CONVEYOR", 16, BELT_Y0 - 8, 0.42, MUTED)

        belt_y = (BELT_Y0 + BELT_Y1) // 2
        x_start, x_end = 60, W - 210
        cv2.line(img, (x_start, belt_y + 22), (x_end, belt_y + 22), LINE, 2)
        for x in range(x_start, x_end, 26):
            cv2.line(img, (x, belt_y + 22), (x + 10, belt_y + 22), DIM, 2)

        if not world.conveyor.running:
            text(img, f"STOPPED: {clip(world.conveyor.stop_reason, 40)}",
                 x_start, BELT_Y0 + 22, 0.44, RED)

        for item in world.conveyor.items:
            x = int(x_start + (x_end - x_start) * item.progress)
            held = item.state is ItemState.HELD
            colour = GREY if held else DESTINATION_COLOUR.get(item.destination, MUTED)
            cv2.rectangle(img, (x - 20, belt_y - 12), (x + 20, belt_y + 16), colour, -1)
            text(img, item.glyph, x - 14, belt_y + 8, 0.42, (20, 20, 20), 1, FD)
            if held:
                text(img, "HELD", x - 14, belt_y - 18, 0.34, AMBER)

        text(img, "->", x_end + 12, belt_y + 8, 0.6, DIM, 1, FD)

    def _bins(self, img, world):
        text(img, "BINS", 16, BIN_Y0 - 8, 0.42, MUTED)
        bins = list(world.bins.bins.values())
        gap, count = 10, len(bins)
        width = (W - 32 - gap * (count - 1)) // count

        for i, bin_ in enumerate(bins):
            x0 = 16 + i * (width + gap)
            x1 = x0 + width
            colour = DESTINATION_COLOUR.get(bin_.destination, MUTED)
            panel(img, x0, BIN_Y0, x1, BIN_Y0 + 116)
            cv2.rectangle(img, (x0, BIN_Y0), (x1, BIN_Y0 + 3), colour, -1)

            text(img, clip(bin_.destination.value.replace("_", " "), 14),
                 x0 + 10, BIN_Y0 + 26, 0.4, TEXT)
            text(img, f"{bin_.percent}%", x0 + 10, BIN_Y0 + 58, 0.7, colour, 1, FD)
            text(img, f"{bin_.current_level} / {bin_.capacity}",
                 x0 + 10, BIN_Y0 + 78, 0.38, DIM)

            bar_y = BIN_Y0 + 92
            cv2.rectangle(img, (x0 + 10, bar_y), (x1 - 10, bar_y + 10), PANEL_2, -1)
            fill = int(x0 + 10 + (width - 20) * min(1.0, bin_.fill_fraction))
            cv2.rectangle(img, (x0 + 10, bar_y), (fill, bar_y + 10), colour, -1)

            if bin_.needs_attention:
                warn = RED if bin_.is_full else AMBER
                text(img, bin_.status, x0 + 10, BIN_Y0 + 112, 0.32, warn)

    def _footer(self, img, pipe):
        panel(img, 16, FOOT_Y0, W - 16, H - 12)
        stats = pipe.stats

        cells = [
            ("PROCESSED", str(stats.processed), TEXT),
            ("SORTED", str(stats.sorted_ok), GREEN),
            ("MANUAL", str(stats.manual_checks), AMBER),
            ("FAILED", str(stats.failures), RED if stats.failures else DIM),
            ("AVG CONF", f"{stats.average_confidence:.0%}", TEXT),
        ]
        x = 30
        for label, value, colour in cells:
            text(img, label, x, FOOT_Y0 + 20, 0.32, DIM)
            text(img, value, x, FOOT_Y0 + 42, 0.54, colour, 1, FD)
            x += 96

        cv2.line(img, (x, FOOT_Y0 + 10), (x, H - 22), LINE, 1)
        alerts = pipe.bus.recent_alerts(limit=2)
        if alerts:
            y = FOOT_Y0 + 22
            for event in reversed(alerts):
                colour = SEVERITY_COLOUR.get(event.severity, MUTED)
                payload = event.to_dict()["payload"]
                detail = payload.get("reason") or payload.get("detail") or event.event
                text(img, f"[{event.severity.value}] {clip(str(detail), 84)}",
                     x + 16, y, 0.38, colour)
                y += 20
        else:
            text(img, "No alerts.", x + 16, FOOT_Y0 + 24, 0.38, DIM)

        text(img, "Q quit    S start/stop belt    E empty bins    F inject failure",
             x + 16, H - 22, 0.34, DIM)
