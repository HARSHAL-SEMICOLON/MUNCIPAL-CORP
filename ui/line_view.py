"""
THE LIVE LINE -- the plant, drawn for a browser.

`simulation/renderer.py` draws the same world with OpenCV, for the desktop
control-room window. This module draws it as SVG instead, because the hosted
demo has no window to open and a browser renders vector text far more sharply
than a pixel buffer scaled into a page.

Nothing here simulates anything. The world, the belt, the bins and the items
are the real `simulation/` classes, and the verdicts driving them come from
the real agents. This file only decides where on the screen each of those
things is drawn.

The layout is deliberately literal: an item travels along the belt toward
**its own bin** rather than to a common end point, so a viewer watching for
ten seconds sees the sort happening -- organics peeling off early, e-waste
carrying on to the far end -- instead of a row of counters ticking up.
"""

from __future__ import annotations

from html import escape

from core.messages import Destination
from simulation.waste_objects import ItemState, WasteItem
from simulation.world import World

# The bins, left to right, in the order a reader will find them every visit.
STREAM_ORDER: list[Destination] = [
    Destination.RECYCLING,
    Destination.ORGANIC,
    Destination.E_WASTE,
    Destination.HAZARDOUS,
    Destination.REJECT,
    Destination.MANUAL_CHECK,
]

# --- canvas geometry -------------------------------------------------------
W, H = 980, 470
BELT_X0, BELT_X1 = 54, 950
BELT_Y = 138                     # top edge of the belt surface
BELT_H = 30
BIN_TOP = 256
BIN_W, BIN_H = 108, 128
DROP_STARTS = 0.86               # progress at which an item leaves the belt

# The bins start well to the right of the belt head. Spacing them across the
# WHOLE belt would put the first bin directly under the loading point, so
# recyclables -- the commonest stream -- would appear to drop in the instant
# they arrived, which reads as a bug rather than as a sort.
BIN_ZONE_X0, BIN_ZONE_X1 = 210, 950
ITEM_START = BELT_X0 + 32


def _bin_centre(index: int) -> float:
    """Screen x of bin `index`, evenly spaced across the bin zone."""
    slot = (BIN_ZONE_X1 - BIN_ZONE_X0) / len(STREAM_ORDER)
    return BIN_ZONE_X0 + slot * (index + 0.5)


def _bin_index(destination: Destination) -> int:
    try:
        return STREAM_ORDER.index(destination)
    except ValueError:
        return STREAM_ORDER.index(Destination.MANUAL_CHECK)


def _item_xy(item: WasteItem) -> tuple[float, float]:
    """Where an item sits: riding the belt, then dropping into its own bin.

    The y returned is the item's BOTTOM edge, and on the belt it sits a few
    pixels into the belt surface so the object reads as resting on it rather
    than hovering above it.
    """
    target = _bin_centre(_bin_index(item.destination))
    riding_y = BELT_Y + 9
    p = item.progress

    if p <= DROP_STARTS:
        travel = p / DROP_STARTS
        x = ITEM_START + (target - ITEM_START) * travel
        return x, riding_y
    # the drop: x holds over the bin, y falls toward its mouth
    fall = (p - DROP_STARTS) / (1.0 - DROP_STARTS)
    return target, riding_y + (BIN_TOP + 22 - riding_y) * fall


# --- pieces ----------------------------------------------------------------

def _belt(tick: int, running: bool, T: dict) -> str:
    """The belt surface, with tread marks that move only when it does."""
    # Scrolling the dash offset is what sells motion; a stopped belt freezes,
    # which is the single clearest signal that a safety rung has fired.
    offset = -(tick * 7) % 28 if running else 0
    tread = T["muted"] if running else T["grid"]
    rail = T["hue_soft"] if running else T["grid"]
    mid = BELT_Y + BELT_H / 2
    return (
        # the belt body, with rails top and bottom so it reads as a machine
        f"<rect x='{BELT_X0}' y='{BELT_Y}' width='{BELT_X1 - BELT_X0}' "
        f"height='{BELT_H}' rx='7' fill='{T['hue_soft']}' fill-opacity='0.45' "
        f"stroke='{rail}' stroke-width='2'/>"
        f"<line x1='{BELT_X0}' y1='{BELT_Y}' x2='{BELT_X1}' y2='{BELT_Y}' "
        f"stroke='{rail}' stroke-width='2.5'/>"
        f"<line x1='{BELT_X0}' y1='{BELT_Y + BELT_H}' x2='{BELT_X1}' "
        f"y2='{BELT_Y + BELT_H}' stroke='{rail}' stroke-width='2.5'/>"
        # tread marks: the motion cue, frozen the instant the belt stops
        f"<line x1='{BELT_X0 + 6}' y1='{mid}' x2='{BELT_X1 - 6}' y2='{mid}' "
        f"stroke='{tread}' stroke-width='2.5' stroke-dasharray='9 19' "
        f"stroke-dashoffset='{offset}' opacity='0.8'/>"
    )


def _decision_line(T: dict) -> str:
    """The commit point, at the belt head.

    Drawn at the START of the belt on purpose: by the time an object is on
    this belt the agent chain has already run, exactly once, which is the
    decision-line rule the pipeline is built around. Putting the marker
    mid-belt would suggest items are still being decided as they travel.
    """
    x = BELT_X0 + 14
    return (
        f"<line x1='{x}' y1='{BELT_Y - 52}' x2='{x}' y2='{BELT_Y + BELT_H + 6}' "
        f"stroke='{T['accent']}' stroke-width='1.5' stroke-dasharray='4 4'/>"
        f"<text x='{x + 7}' y='{BELT_Y - 56}' font-size='9.5' "
        f"fill='{T['accent']}' font-family='monospace'>"
        f"DECISION POINT &#183; ONE COMMIT PER OBJECT</text>"
    )


def _item(item: WasteItem, T: dict, streams: dict, status: dict) -> str:
    x, y = _item_xy(item)
    colour = streams.get(item.destination.value, T["muted"])
    held = item.state is ItemState.HELD
    stroke = status["warning"] if held else colour
    w, h = 48, 30

    out = (
        f"<g transform='translate({x - w / 2:.1f},{y - h:.1f})'>"
        f"<rect width='{w}' height='{h}' rx='5' fill='{colour}' "
        f"fill-opacity='{0.25 if held else 0.9}' stroke='{stroke}' "
        f"stroke-width='{2 if held else 1}'/>"
        f"<text x='{w / 2}' y='{h / 2 + 4}' font-size='11' text-anchor='middle' "
        f"font-family='monospace' font-weight='600' "
        f"fill='{T['ink'] if held else T['surface']}'>"
        f"{escape(item.glyph)}</text></g>"
    )
    if held:
        out += (
            f"<text x='{x}' y='{y - h - 6}' font-size='9' text-anchor='middle' "
            f"font-family='monospace' fill='{status['warning']}'>HELD</text>"
        )
    return out


def _diverter(item: WasteItem | None, T: dict, streams: dict) -> str:
    """The arm, drawn engaged only while an item is actually dropping."""
    if item is None:
        return ""
    x, _ = _item_xy(item)
    colour = streams.get(item.destination.value, T["accent"])
    top = BELT_Y + BELT_H
    return (
        # the arm, swung down out of the belt toward the bin it is feeding
        f"<line x1='{x - 26}' y1='{top + 2}' x2='{x + 4}' y2='{top + 20}' "
        f"stroke='{colour}' stroke-width='5' stroke-linecap='round'/>"
        # the chute it is pushing into
        f"<line x1='{x}' y1='{top + 20}' x2='{x}' y2='{BIN_TOP - 4}' "
        f"stroke='{colour}' stroke-width='1.5' stroke-dasharray='3 5' "
        f"opacity='0.7'/>"
    )


def _bin(index: int, bin_, T: dict, streams: dict, status: dict,
         flash: bool) -> str:
    cx = _bin_centre(index)
    x0 = cx - BIN_W / 2
    colour = streams.get(bin_.destination.value, T["muted"])

    if bin_.is_full:
        edge, edge_w = status["critical"], 2.5
    elif bin_.needs_attention:
        edge, edge_w = status["warning"], 2.0
    elif flash:
        edge, edge_w = colour, 2.5
    else:
        edge, edge_w = T["grid"], 1.0

    fill_h = max(0.0, min(1.0, bin_.fill_fraction)) * (BIN_H - 8)
    name = bin_.destination.value.replace("_", " ")

    out = (
        f"<rect x='{x0}' y='{BIN_TOP}' width='{BIN_W}' height='{BIN_H}' rx='7' "
        f"fill='{T['card']}' stroke='{edge}' stroke-width='{edge_w}'/>"
        f"<rect x='{x0 + 4}' y='{BIN_TOP + BIN_H - 4 - fill_h:.1f}' "
        f"width='{BIN_W - 8}' height='{fill_h:.1f}' rx='4' fill='{colour}' "
        f"fill-opacity='0.85'/>"
        f"<text x='{cx}' y='{BIN_TOP - 10}' font-size='11' text-anchor='middle' "
        f"font-family='monospace' fill='{colour}' font-weight='600'>"
        f"{escape(name)}</text>"
        f"<text x='{cx}' y='{BIN_TOP + BIN_H + 18}' font-size='15' "
        f"text-anchor='middle' font-weight='700' fill='{T['ink']}'>"
        f"{bin_.percent}%</text>"
        f"<text x='{cx}' y='{BIN_TOP + BIN_H + 33}' font-size='10' "
        f"text-anchor='middle' font-family='monospace' fill='{T['muted']}'>"
        f"{bin_.current_level}/{bin_.capacity}</text>"
    )
    if bin_.needs_attention:
        warn = status["critical"] if bin_.is_full else status["warning"]
        out += (
            f"<text x='{cx}' y='{BIN_TOP + BIN_H + 47}' font-size='9' "
            f"text-anchor='middle' font-family='monospace' fill='{warn}'>"
            f"{escape(bin_.status)}</text>"
        )
    return out


def _banner(text: str, colour: str, T: dict) -> str:
    """The interlock notice. Only drawn when a rung has actually stopped it."""
    return (
        f"<rect x='{BELT_X0}' y='34' width='{BELT_X1 - BELT_X0}' height='32' "
        f"rx='6' fill='{colour}' fill-opacity='0.16' stroke='{colour}'/>"
        f"<text x='{W / 2}' y='55' font-size='12.5' text-anchor='middle' "
        f"font-family='monospace' font-weight='600' fill='{colour}'>"
        f"{escape(text)}</text>"
    )


def _status_strip(world: World, T: dict, status: dict) -> str:
    running = world.conveyor.running
    colour = status["good"] if running else status["critical"]
    word = "BELT RUNNING" if running else "BELT STOPPED"
    on_belt = len(world.conveyor.items)
    held = len(world.conveyor.held_items)

    return (
        f"<circle cx='{BELT_X0 + 6}' cy='16' r='5' fill='{colour}'/>"
        f"<text x='{BELT_X0 + 18}' y='20' font-size='12' font-family='monospace' "
        f"font-weight='600' fill='{colour}'>{word}</text>"
        f"<text x='{BELT_X1}' y='20' font-size='12' text-anchor='end' "
        f"font-family='monospace' fill='{T['muted']}'>"
        f"sorted {world.bins.total_sorted} &#183; on belt {on_belt} &#183; "
        f"held {held}</text>"
    )


# --- the frame -------------------------------------------------------------

def frame(world: World, T: dict, streams: dict, status: dict, *,
          tick: int = 0, flash: dict | None = None, banner: str = "",
          banner_kind: str = "critical") -> str:
    """One complete frame of the line, as an inline SVG string."""
    flash = flash or {}
    parts = [
        f"<svg viewBox='0 0 {W} {H}' width='100%' "
        f"xmlns='http://www.w3.org/2000/svg' "
        f"style='font-family:system-ui,sans-serif'>",
        f"<rect width='{W}' height='{H}' fill='{T['surface']}'/>",
        _status_strip(world, T, status),
    ]

    if banner:
        parts.append(_banner(banner, status.get(banner_kind, status["critical"]), T))

    parts.append(_belt(tick, world.conveyor.running, T))
    parts.append(_decision_line(T))

    # The arm engages for whichever item is mid-drop this frame.
    dropping = next((i for i in world.conveyor.items
                     if i.state is ItemState.TRAVELLING
                     and i.progress > DROP_STARTS), None)
    parts.append(_diverter(dropping, T, streams))

    for index, destination in enumerate(STREAM_ORDER):
        bin_ = world.bins.get(destination)
        if bin_ is not None:
            parts.append(_bin(index, bin_, T, streams, status,
                              flash.get(destination.value, 0) > 0))

    for item in world.conveyor.items:
        parts.append(_item(item, T, streams, status))

    parts.append("</svg>")
    return "".join(parts)
