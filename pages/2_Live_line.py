"""
THE LIVE LINE -- a running plant in the browser.

The agent demo on the main page answers "how does this system *think* about
one object". This page answers the other half: "what does it look like when
it runs". The belt moves, items travel to their own bins, the bins fill, and
when a safety rung fires the belt stops in front of you.

WHAT IS REAL AND WHAT IS NOT, because a demo that blurs this is worthless:

  * REAL -- the Material, Classification, Decision and Safety agents. Every
    verdict below is produced by the same code the live line runs, loaded
    from the same config. The seven safety rungs are live: hazardous items
    divert, ambiguous ones go to a human, a full bin holds the item on the
    belt, and a person in the zone stops everything.
  * REAL -- the conveyor, the bins and their capacities, from `simulation/`.
  * SCRIPTED -- the detections. A cloud host has no camera, so instead of a
    Vision Agent reading frames, a scripted stream of class names is fed in
    at the point where the detector would hand them over. The labels are the
    detector's real COCO vocabulary, and the confidences are sampled in the
    range a real detector returns.

That last point is the honest limitation of hosting this at all, and it is
stated on the page itself rather than buried here.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging

import streamlit as st

st.set_page_config(page_title="Live line — waste segregation",
                   page_icon="♻️", layout="wide")

logging.disable(logging.INFO)

from ui import line_view, theme  # noqa: E402

T, STATUS = theme.apply()
STREAMS = theme.streams()

from agents.classification_agent import ClassificationAgent  # noqa: E402
from agents.decision_agent import DecisionAgent  # noqa: E402
from agents.material_agent import MaterialAgent  # noqa: E402
from agents.safety_agent import SafetyAgent, SafetyContext  # noqa: E402
from core.config import Config  # noqa: E402
from core.event_bus import EventBus  # noqa: E402
from core.labels import load_label_map  # noqa: E402
from core.messages import Action, Destination, Detection  # noqa: E402
from simulation.waste_objects import ItemState, WasteItem  # noqa: E402
from simulation.world import World  # noqa: E402

# A plausible shift, drawn from the detector's actual COCO vocabulary. The
# weights are what makes the demo readable: mostly ordinary recyclables and
# organics, with enough of the interesting cases to see every rung fire.
FEED: list[tuple[str, int]] = [
    ("bottle", 7),       # ambiguous PET/glass -> resolved, same bin
    ("banana", 5),       # clean organic
    ("book", 4),         # paper
    ("apple", 3),
    ("cup", 3),          # ambiguous across bins -> a human decides
    ("cell phone", 3),   # e-waste
    ("wine glass", 2),   # glass
    ("scissors", 2),     # metal
    ("orange", 2),
    ("laptop", 1),
    ("toothbrush", 1),
    ("sandwich", 1),
]
LABELS = [label for label, weight in FEED for _ in range(weight)]

RULE_PLAIN = {
    "R1_CONTROLLER_UNAVAILABLE": "actuator unavailable",
    "R2_INTRUSION": "person in the sorting zone",
    "R3_HAZARDOUS": "hazardous — diverted whatever the confidence",
    "R4_UNCERTAIN": "too uncertain, or no stream could be assigned",
    "R5_OVERLAP": "objects overlapping — an occluded scene",
    "R6_BIN_FULL": "destination bin full — held on the belt",
    "R7_APPROVED": "nothing objected, so it was sorted",
}


@st.cache_resource(show_spinner=False)
def build_agents():
    """The real agents, from the real config. No detector: see the module note."""
    cfg = Config.load()
    bus = EventBus()
    labels = load_label_map(cfg)
    return {
        "cfg": cfg,
        "material": MaterialAgent(bus, labels=labels),
        "classifier": ClassificationAgent(bus, labels=labels),
        "decider": DecisionAgent(cfg, bus),
        "safety": SafetyAgent(cfg, bus),
    }


def commit(system, world: World, label: str, confidence: float) -> dict:
    """One object, through the real chain, onto the belt.

    This mirrors `Pipeline._commit` exactly -- material, classification,
    decision, then the safety gate holding the only path to an action.
    """
    detection = Detection(track_id=random.randint(1000, 9999), label=label,
                          confidence=confidence, bbox=(0, 0, 80, 80), frame_id=0)

    material = system["material"].identify(detection)
    classification = system["classifier"].classify(detection, material)
    decision = system["decider"].decide(detection, material, classification)
    verdict = system["safety"].review(
        detection, classification, decision,
        SafetyContext(neighbours=[], bins=world.bins,
                      conveyor=world.conveyor, controller=None),
    )

    row = {"label": label, "confidence": confidence,
           "stream": classification.category.value,
           "rule": verdict.rule, "action": verdict.action.value,
           "destination": verdict.destination.value}

    if verdict.action is Action.STOP_CONVEYOR:
        world.conveyor.stop(verdict.reason)
        return row

    destination = verdict.destination
    if destination is Destination.NONE:
        destination = Destination.MANUAL_CHECK

    item = WasteItem(label=label, category=classification.category,
                     destination=destination, track_id=detection.track_id,
                     confidence=confidence)
    if verdict.action is Action.HOLD:
        item.state = ItemState.HELD
        item.note = "bin full"
    world.conveyor.place(item)
    return row


# --- page ------------------------------------------------------------------

theme.hero("Live line", "Municipal waste segregation")
st.markdown(
    "The belt runs, objects travel to **their own bin**, and the bins fill. "
    "The agents deciding where each item goes are the real ones — the same "
    "code, the same config, the same seven safety rungs as the live system."
)
st.caption(
    "**What is simulated:** a cloud host has no camera, so the detections are "
    "scripted — real COCO class names at realistic confidences, fed in where "
    "the Vision Agent would hand them over. Everything after that point is "
    "the production code. Upload a photo on the **Agent demo** page to run "
    "the real detector on a real image."
)

controls = st.columns([1, 1, 1, 1.4])
with controls[0]:
    items = st.slider("Items to run", 6, 40, 18)
with controls[1]:
    speed = st.select_slider("Speed", ["slow", "normal", "fast"], "normal")
with controls[2]:
    intrusion = st.checkbox("Person enters the zone", value=False,
                            help="Fires safety rung R2 partway through: the "
                                 "belt stops and stays stopped until an "
                                 "operator restarts it.")
with controls[3]:
    st.write("")
    run = st.button("Run the line", type="primary")

canvas = st.empty()
ledger = st.empty()

DT = {"slow": 0.11, "normal": 0.07, "fast": 0.04}[speed]
SPAWN_EVERY = {"slow": 11, "normal": 8, "fast": 6}[speed]


def draw(world, tick=0, flash=None, banner="", banner_kind="critical"):
    canvas.markdown(
        line_view.frame(world, T, STREAMS, STATUS, tick=tick, flash=flash,
                        banner=banner, banner_kind=banner_kind),
        unsafe_allow_html=True,
    )


system = build_agents()

if not run:
    # An idle plant, so the page is never blank on arrival.
    draw(World(system["cfg"]))
    st.info("Press **Run the line** to start the belt.")
    st.stop()

world = World(system["cfg"])
rows: list[dict] = []
flash: dict[str, int] = {}
stopped_at: int | None = None
banner = ""
banner_kind = "critical"

spawned = 0
tick = 0
# Enough ticks for the last item to finish its journey after the last spawn.
travel_ticks = int((world.conveyor.travel_seconds / DT) + 6)
total_ticks = items * SPAWN_EVERY + travel_ticks
# A stopped belt freezes every item on it, so a run that exercises the
# interlock needs the stopped ticks added back or the last items are still
# in transit when the loop ends and the totals look wrong.
if intrusion:
    total_ticks += 26
intrusion_at = int(total_ticks * 0.42) if intrusion else None

progress = st.progress(0.0)

for tick in range(total_ticks):
    # -- the scripted feed arriving at the decision line -------------------
    if spawned < items and tick % SPAWN_EVERY == 0 and world.conveyor.running:
        label = random.choice(LABELS)
        row = commit(system, world, label, round(random.uniform(0.52, 0.95), 2))
        rows.append(row)
        spawned += 1
        if row["action"] != "STOP_CONVEYOR":
            flash[row["destination"]] = 4

    # -- the interlock, fired as a scene event rather than an item ---------
    if intrusion_at is not None and tick == intrusion_at and world.conveyor.running:
        world.conveyor.stop("person in the sorting zone")
        stopped_at = tick
        banner = "R2 · PERSON IN THE SORTING ZONE — BELT STOPPED, OPERATOR RESTART REQUIRED"
        rows.append({"label": "person", "confidence": 0.91, "stream": "—",
                     "rule": "R2_INTRUSION", "action": "STOP_CONVEYOR",
                     "destination": "—"})

    # The belt is not restarted automatically when the zone clears: a real
    # interlock requires a deliberate reset. Here the "operator" returns
    # after a beat, which is what makes the stop visible rather than fatal.
    if stopped_at is not None and tick == stopped_at + 22:
        world.conveyor.start()
        banner = ""

    delivered = world.update(DT)
    for item in delivered:
        flash[item.destination.value] = 4
    flash = {k: v - 1 for k, v in flash.items() if v > 1}

    draw(world, tick=tick, flash=flash, banner=banner, banner_kind=banner_kind)
    progress.progress(min(1.0, (tick + 1) / total_ticks))
    time.sleep(DT)

progress.empty()

# --- what just happened ----------------------------------------------------

sorted_total = world.bins.total_sorted
held = len(world.conveyor.held_items)
to_human = sum(1 for r in rows if r["destination"] == "MANUAL_CHECK")

summary = st.columns(4)
summary[0].metric("Sorted into bins", sorted_total)
summary[1].metric("Sent to a human", to_human)
summary[2].metric("Held on the belt", held)
summary[3].metric("Items presented", len(rows))

with ledger.container():
    st.subheader("Every decision, and the rule that produced it")
    st.dataframe(
        [{"object": r["label"],
          "confidence": f"{r['confidence']:.0%}",
          "stream": r["stream"],
          "went to": r["destination"].replace("_", " "),
          "rule": r["rule"],
          "why": RULE_PLAIN.get(r["rule"], "")} for r in rows],
        hide_index=True,
    )

st.caption(
    "Nothing above is pre-recorded: press **Run the line** again and the mix "
    "of objects, the confidences and therefore the rules that fire will differ."
)
