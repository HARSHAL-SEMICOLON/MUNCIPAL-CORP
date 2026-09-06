"""
The deployable demo: upload an object, watch the agents decide.

    streamlit run streamlit_app.py

This is the entry point a hosted deployment serves, and it is deliberately
NOT the live sorting line. That loop needs a webcam and a desktop window,
neither of which a cloud host has -- and the shift report needs a database
that is gitignored, so a deployed copy would show an empty page.

What travels well is the part that makes this project interesting: the
reasoning. One image in, and every agent's output shown in order, including
the ones that say "I cannot tell" and the rule that decided what to do about
it. A visitor sees the whole chain in about four seconds.

The agents here are the real ones, loaded from the same config as the live
system. Nothing is mocked and no answer is pre-baked.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import logging

import numpy as np
import streamlit as st

st.set_page_config(page_title="Waste Segregation — Agent Demo",
                   page_icon="♻️", layout="wide")

logging.disable(logging.INFO)

# --- palette (matches the reporting view) ---------------------------------
LIGHT = {"hue": "#2a78d6", "ink": "#0b0b0b", "muted": "#52514e",
         "grid": "#e6e5e1", "card": "#f6f6f4"}
DARK = {"hue": "#3987e5", "ink": "#ffffff", "muted": "#c3c2b7",
        "grid": "#302f2d", "card": "#1e211f"}
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}
STREAM_COLOUR = {
    "RECYCLING": "#2a78d6", "ORGANIC": "#008300", "E_WASTE": "#4a3aa7",
    "HAZARDOUS": "#e34948", "REJECT": "#8a8880", "MANUAL_CHECK": "#eda100",
}

RULE_PLAIN = {
    "R1_CONTROLLER_UNAVAILABLE": "the actuator was unavailable",
    "R2_INTRUSION": "a person or animal was in the sorting zone",
    "R3_HAZARDOUS": "it is a hazardous stream — diverted whatever the confidence",
    "R4_UNCERTAIN": "too uncertain, or no stream could be assigned",
    "R5_OVERLAP": "objects were overlapping — an occluded scene",
    "R6_BIN_FULL": "the destination bin was full",
    "R7_APPROVED": "nothing objected, so it was sorted",
}


def theme() -> dict:
    try:
        if st.context.theme.type == "dark":
            return DARK
    except Exception:
        pass
    return LIGHT


T = theme()


# --- the real system, loaded once -----------------------------------------

@st.cache_resource(show_spinner="Loading the detector and agents…")
def load_system():
    """Real agents, real weights, same config as the live line.

    The weights are not in the repository -- binaries do not belong in git --
    so on a fresh host they are fetched once and cached. Everything else
    comes straight from the clone.
    """
    from ultralytics import YOLO

    from agents.classification_agent import ClassificationAgent
    from agents.decision_agent import DecisionAgent
    from agents.material_agent import MaterialAgent
    from agents.routing_agent import RoutingAgent
    from agents.safety_agent import SafetyAgent
    from core.config import Config
    from core.event_bus import EventBus
    from core.labels import load_label_map
    from vision.detector import Detector

    cfg = Config.load()
    weights = cfg.path("detection.model_path", "models/yolov8n.pt")
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        YOLO("yolov8n.pt")                       # ultralytics fetches to cwd
        for candidate in (ROOT / "yolov8n.pt", Path("yolov8n.pt")):
            if candidate.exists():
                candidate.replace(weights)
                break

    bus = EventBus()
    labels = load_label_map(cfg)
    return {
        "cfg": cfg,
        "labels": labels,
        "detector": Detector(cfg),
        "material": MaterialAgent(bus, labels=labels),
        "classifier": ClassificationAgent(bus, labels=labels),
        "decider": DecisionAgent(cfg, bus),
        "safety": SafetyAgent(cfg, bus),
        "routing": RoutingAgent(cfg, bus),
    }


def decide(system, image_bgr):
    """Run the real chain over one image and return every step."""
    from agents.safety_agent import SafetyContext
    from core.routing import destination_for

    detections = system["detector"].detect(image_bgr, frame_id=1)
    if not detections:
        return {"detected": False, "detections": []}

    det = max(detections, key=lambda d: (d.confidence, d.area))
    material = system["material"].identify(det)
    classification = system["classifier"].classify(det, material)
    decision = system["decider"].decide(det, material, classification)
    verdict = system["safety"].review(det, classification, decision,
                                      SafetyContext(neighbours=detections))
    route = system["routing"].route_for(
        classification.category if verdict.destination.value not in
        ("MANUAL_CHECK", "HAZARDOUS") else classification.category)

    return {
        "detected": True, "detections": detections, "det": det,
        "material": material, "classification": classification,
        "decision": decision, "verdict": verdict, "route": route,
        "expected_bin": destination_for(classification.category).value,
    }


def annotate(image_bgr, detections, chosen):
    import cv2
    view = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        main = d is chosen
        colour = (214, 120, 42) if main else (150, 150, 150)
        cv2.rectangle(view, (x1, y1), (x2, y2), colour, 3 if main else 1)
        tag = f"{d.label} {d.confidence:.0%}"
        cv2.putText(view, tag, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def step(n: int, agent: str, question: str, answer: str, detail: str = "",
         colour: str | None = None) -> None:
    colour = colour or T["ink"]
    st.markdown(
        f"<div style='padding:11px 0;border-bottom:1px solid {T['grid']}'>"
        f"<span style='font-family:monospace;font-size:11px;color:{T['muted']}'>"
        f"{n} · {agent.upper()}</span><br>"
        f"<span style='font-size:12.5px;color:{T['muted']}'>{question}</span><br>"
        f"<span style='font-size:17px;font-weight:600;color:{colour}'>{answer}</span>"
        + (f"<br><span style='font-size:13px;color:{T['muted']}'>{detail}</span>"
           if detail else "")
        + "</div>", unsafe_allow_html=True)


# --- page ------------------------------------------------------------------

st.title("Waste Segregation — Agent Demo")
st.markdown(
    "A camera watches a conveyor and ten agents decide what each object is, "
    "which waste stream it belongs to, whether it is **safe to act on that "
    "conclusion**, and where it goes. Upload a photo of an object and watch "
    "the chain run.\n\n"
    "The agents below are the real ones, loaded from the same configuration "
    "as the live system. Nothing is mocked."
)

samples = sorted((ROOT / "dashboard" / "samples").glob("*.jpg")) \
    if (ROOT / "dashboard" / "samples").exists() else []

left, right = st.columns([1, 1])

with left:
    st.subheader("An object")
    uploaded = st.file_uploader("Photograph of a single object",
                                type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    picked = None
    if samples:
        names = ["—"] + [p.stem.replace("_", " ") for p in samples]
        choice = st.selectbox("or try a sample", names)
        if choice != "—":
            picked = samples[names.index(choice) - 1]

    st.caption(
        "Works today on: bottle, banana, apple, orange, book, cup, bowl, "
        "phone, laptop, scissors, cutlery, wine glass. "
        "**Battery, aluminium can, cardboard and plastic bag are not classes "
        "the current model was trained on** — try one anyway and watch the "
        "system say so rather than guess."
    )

source = uploaded if uploaded is not None else picked

if source is None:
    st.info("Upload an image to run the agents.")
    st.stop()

import cv2

if uploaded is not None:
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
else:
    image = cv2.imread(str(picked))

if image is None:
    st.error("That file could not be read as an image.")
    st.stop()

system = load_system()
result = decide(system, image)

with left:
    if result["detected"]:
        st.image(annotate(image, result["detections"], result["det"]),
                 use_container_width=True)
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

with right:
    st.subheader("What the agents concluded")

    if not result["detected"]:
        step(1, "Vision Agent", "Is there an object, and where?",
             "Nothing detected", colour=STATUS["warning"])
        st.warning(
            "**This is a correct outcome, not a crash.** The detector is "
            "COCO-pretrained and has never been trained on this object, so it "
            "reports nothing rather than inventing a label. On a real line the "
            "item would travel on to manual inspection."
        )
        st.stop()

    det = result["det"]
    material = result["material"]
    classification = result["classification"]
    verdict = result["verdict"]
    dest = verdict.destination.value
    colour = STREAM_COLOUR.get(dest, T["ink"])

    step(1, "Vision Agent", "Is there an object, and where?",
         f"{det.label}", f"{det.confidence:.0%} confidence")

    candidates = " or ".join(c.value for c in material.candidates)
    step(2, "Material Agent", "What is it made of?",
         material.material.value,
         (f"could be {candidates} — the detector's class does not separate them"
          if material.candidates else material.reason),
         colour=STATUS["warning"] if material.material.value == "UNKNOWN" else None)

    step(3, "Classification Agent", "Which waste stream?",
         classification.category.value, classification.reason)

    step(4, "Decision Agent", "What ought to happen? (a proposal, not a command)",
         f"{result['decision'].action.value} → {result['decision'].destination.value}",
         result["decision"].reason)

    approved = verdict.approved
    step(5, "Safety Agent", "Is that safe to do?",
         "Approved" if approved else f"Overridden → {verdict.action.value}",
         f"<b>{verdict.rule}</b> — {RULE_PLAIN.get(verdict.rule, verdict.reason)}",
         colour=STATUS["good"] if approved else STATUS["serious"])

    st.markdown(
        f"<div style='margin-top:18px;padding:16px 18px;background:{T['card']};"
        f"border-left:4px solid {colour}'>"
        f"<span style='font-family:monospace;font-size:11px;color:{T['muted']};"
        f"letter-spacing:.12em'>OUTCOME</span><br>"
        f"<span style='font-size:24px;font-weight:700;color:{colour}'>"
        f"{dest.replace('_', ' ')}</span><br>"
        f"<span style='font-size:13.5px;color:{T['muted']}'>{verdict.reason}</span>"
        "</div>", unsafe_allow_html=True)

    route = result["route"]
    if route is not None:
        st.caption(f"**Downstream:** {route.stream} → {route.chain}. "
                   "No specific facility is named — the system describes the "
                   "kind of destination and will not invent an organisation.")

st.divider()

with st.expander("Why this is more than a classifier"):
    st.markdown(
        """
A classifier is `image → label`. It answers one question and stops: it cannot
tell you why, and it cannot decline to answer.

Three things above are doing something else:

**The Material Agent is allowed to say UNKNOWN.** A COCO detector reports
`bottle` without distinguishing PET from glass — both were the same class in
its training data. Guessing would put glass in the plastics stream and look
confident doing it.

**The Classification Agent asks whether the ambiguity matters.** PET and glass
are both dry recyclables headed for the same bin, so a bottle proceeds. A cup
might be plastic, paper or ceramic — which span *different* bins — so it goes
to a human. Same uncertainty, two different answers, because the consequence
differs.

**The Decision Agent never commands anything.** It proposes; the Safety Agent
holds the only path to actuation and applies seven ordered rules. Six of the
seven stop, hold or divert — sorting is the last resort, not the default.

That last point is the clearest test: **a classifier always answers, and this
system can refuse.**
        """
    )
