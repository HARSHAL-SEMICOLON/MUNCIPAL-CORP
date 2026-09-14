"""
LIVE CAMERA -- the real loop, streamed into the browser.

This is `main.py` with the OpenCV window replaced by a WebRTC video track.
Everything between the camera and the screen is the production path, not a
demo path:

    frame -> VisionAgent (YOLO + ByteTrack)
          -> the decision line (commit each object exactly ONCE)
          -> Material -> Classification -> Decision -> Safety
          -> ActionAgent -> SimulationController -> conveyor + bins
          -> Renderer (the same control-room view the desktop app draws)

Where the camera lives is the only thing that changed. `main.py` opens a
webcam on the machine running the code; a cloud host has none, so here the
browser captures the stream and WebRTC carries it to the server, which
processes each frame and sends back the rendered control room. The pipeline
cannot tell the difference.

ONE HONEST NOTE ABOUT SPEED. Every frame makes a round trip to the server
and back, and inference on a free shared CPU is roughly 100 ms. Expect a
handful of frames per second, not thirty. `python main.py` on your own
machine is smoother because nothing leaves it. What this page buys is that
anyone can open a link and hold a bottle up to their own camera.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging

import streamlit as st

st.set_page_config(page_title="Live camera — waste segregation",
                   page_icon="♻️", layout="wide")

logging.disable(logging.INFO)

import av  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

from ui import theme  # noqa: E402

T, STATUS = theme.apply()

from core.config import Config  # noqa: E402
from core.event_bus import EventBus  # noqa: E402
from hardware.controller import ControllerUnavailable, build_controller  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from simulation.renderer import Renderer  # noqa: E402
from simulation.world import World  # noqa: E402
from vision.detector import ModelUnavailable  # noqa: E402

STUN_ONLY = [{"urls": ["stun:stun.l.google.com:19302"]}]


def ice_servers() -> tuple[list[dict], bool]:
    """ICE servers for the connection, and whether a TURN relay is among them.

    THE THING THAT DECIDES WHETHER THIS PAGE WORKS AT ALL.

    STUN only tells each side what its own public address is; the two then
    talk directly. That works on a home network and fails on Streamlit
    Community Cloud, because the app sits behind a proxy that will not pass
    the direct media path -- the handshake completes, no frames ever arrive,
    and the viewer sees a black rectangle with no error, which is the worst
    failure mode a page can have.

    A TURN server fixes it by relaying the media instead of merely describing
    the route. It costs bandwidth, so it is never free-and-anonymous for long;
    the credentials belong in Streamlit secrets rather than in this file:

        # .streamlit/secrets.toml  (or the Secrets box on Streamlit Cloud)
        [turn]
        urls = ["turn:standard.relay.metered.ca:80"]
        username = "..."
        credential = "..."

    Free tiers that work: metered.ca's Open Relay (20 GB/month) and Twilio's
    Network Traversal Service (trial credit). Without them this page falls
    back to STUN and says so on screen rather than pretending.
    """
    try:
        turn = st.secrets.get("turn")
    except Exception:            # no secrets file at all, which is normal
        turn = None

    if turn and turn.get("urls"):
        server = {"urls": list(turn["urls"])}
        if turn.get("username"):
            server["username"] = turn["username"]
        if turn.get("credential"):
            server["credential"] = turn["credential"]
        return STUN_ONLY + [server], True

    return STUN_ONLY, False


class Plant:
    """One viewer's line: world, pipeline and renderer, built once.

    Kept in session state rather than a cached resource so two people opening
    the demo at the same time do not pour waste into each other's bins.
    """

    def __init__(self):
        self.cfg = Config.load()
        self.bus = EventBus()
        self.world = World(self.cfg)
        self.controller = build_controller(self.cfg, self.world)
        self.pipe = Pipeline(self.cfg, self.bus, self.world, self.controller,
                             db=None)
        self.renderer = Renderer(self.cfg)
        self.frame_id = 0
        self.last_tick = time.monotonic()
        # Boxes from the most recent frame that was actually run through the
        # detector, redrawn on the frames in between (see `process_every`).
        self.last_detections: list = []
        # Straight from config, so restoring it after the toggle below cannot
        # drift from what the live system actually runs.
        self.intrusion_classes = set(self.pipe.safety.intrusion_classes)


def get_plant() -> Plant | None:
    if "plant" not in st.session_state:
        try:
            st.session_state.plant = Plant()
        except ModelUnavailable as exc:
            st.error(f"Detection model unavailable: {exc}")
            return None
        except ControllerUnavailable as exc:
            st.error(f"Actuation layer unavailable: {exc}")
            return None
    return st.session_state.plant


# --- page ------------------------------------------------------------------

theme.hero("Live camera", "Municipal waste segregation")
st.markdown(
    "The **whole loop**, running on your own camera: objects are detected and "
    "tracked frame to frame, each one is committed exactly once as it crosses "
    "the decision line, the agents decide where it goes, and it travels the "
    "belt into a bin. This is `main.py` — the same pipeline, the same "
    "renderer — with the desktop window swapped for a video stream."
)

plant = get_plant()
if plant is None:
    st.stop()

controls = st.columns([1, 1, 1, 1.3])
with controls[0]:
    view = st.radio("View", ["Camera only", "Control room"],
                    help="Camera only is the annotated video and is much "
                         "lighter — start here. Control room adds the full "
                         "desktop layout (readout, belt, bins) and costs a "
                         "1024x720 render on every frame.")
with controls[1]:
    size = st.select_slider(
        "Detection size", [320, 416, 640], value=320,
        help="Smaller is faster and less accurate. 640 is what the live "
             "system uses; 320 keeps a slow connection watchable.")
with controls[2]:
    process_every = st.select_slider(
        "Detect every N frames", [1, 2, 3, 4], value=2,
        help="Inference is the expensive step, not the video. Running it on "
             "every second or third frame roughly doubles or triples the "
             "frame rate; the boxes from the last detected frame are redrawn "
             "in between, so motion still looks continuous.")
with controls[3]:
    intrusion_live = st.checkbox(
        "Person in frame stops the belt (rung R2)", value=False,
        help="OFF by default for an obvious reason: you are sitting in front "
             "of your own camera, so the interlock would fire on you and the "
             "belt would never run. Turn it ON to watch R2 work — the belt "
             "stops the moment a person is seen and stays stopped.")

plant.pipe.vision.detector.imgsz = int(size)
# The interlock is a real rung, so it is enabled and disabled by editing the
# class list it watches -- not by branching around the Safety Agent.
plant.pipe.safety.intrusion_classes = (
    set(plant.intrusion_classes) if intrusion_live else set())

if st.button("Reset the line (empty bins, clear the belt)"):
    st.session_state.pop("plant", None)
    st.rerun()


def annotate_camera(img, detections, pipe) -> np.ndarray:
    """The camera view alone: boxes, ids and the decision line."""
    out = img.copy()
    h, w = out.shape[:2]
    line_x = int(w * pipe.line_position)
    cv2.line(out, (line_x, 0), (line_x, h), (96, 116, 206), 2)
    cv2.putText(out, "DECISION LINE", (line_x + 6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (96, 116, 206), 1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        done = pipe.committed(det.track_id)
        colour = (104, 176, 92) if done else (176, 152, 46)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        tag = f"{det.label} {det.confidence:.0%}"
        if done:
            tag += " committed"
        cv2.putText(out, tag, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    record = pipe.last_record
    if record is not None:
        cv2.putText(out, f"{record.label} -> {record.destination.value}",
                    (12, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (234, 233, 230), 1, cv2.LINE_AA)
    return out


def make_callback(plant: Plant, control_room: bool, process_every: int):
    """Build the per-frame callback. Runs on WebRTC's own thread.

    It touches nothing from Streamlit -- only plain objects captured here --
    because widget state is not safe to read from that thread.
    """

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        plant.frame_id += 1
        now = time.monotonic()
        dt = min(0.25, now - plant.last_tick)   # clamp, so a stall cannot
        plant.last_tick = now                   # teleport items down the belt

        try:
            # Inference is the whole cost. The belt still advances by real
            # elapsed time on every frame, so skipping detection changes how
            # often objects are SEEN, never how fast they travel.
            if plant.frame_id % process_every == 0:
                detections = plant.pipe.process_frame(img, plant.frame_id,
                                                      img.shape[1])
                plant.last_detections = detections
            else:
                detections = plant.last_detections
            plant.world.update(dt)
            if control_room:
                out = plant.renderer.draw(img, detections, plant.pipe,
                                          plant.world)
                # The renderer targets a 1280x900 desktop window. Every frame
                # of that goes back over the wire, so it is scaled down once
                # here rather than paid for on every connection.
                out = cv2.resize(out, (1024, 720), interpolation=cv2.INTER_AREA)
            else:
                out = annotate_camera(img, detections, plant.pipe)
        except Exception as exc:  # noqa: BLE001
            # A dropped frame must not kill the stream, but it must be
            # visible rather than silently blank -- the lesson from the
            # tracker dependency that spent a week disguised as "no object".
            out = img.copy()
            cv2.putText(out, f"frame error: {type(exc).__name__}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 62, 214), 2,
                        cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    return callback


servers, has_turn = ice_servers()

if not has_turn:
    st.warning(
        "**No TURN server configured — on Streamlit Community Cloud the video "
        "will very likely stay black.** The app runs behind a proxy that "
        "blocks the direct browser↔server media path, so a relay is required. "
        "This is a hosting limitation, not a fault in the pipeline: the same "
        "code runs fine locally. See *Getting the video to connect* below."
    )

ctx = webrtc_streamer(
    key="waste-line",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": servers},
    video_frame_callback=make_callback(plant, view == "Control room",
                                       int(process_every)),
    media_stream_constraints={
        # 640x480 keeps the round trip affordable; the detector downscales to
        # `imgsz` anyway, so a larger capture buys nothing but latency.
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False,
    },
    async_processing=True,
)

if not ctx.state.playing:
    st.info(
        "Press **START** above and allow camera access. Then hold an object "
        "up and move it slowly across the frame, left to right — it is "
        "committed as it crosses the decision line."
    )

with st.expander("Getting the video to connect (read this if it stays black)"):
    st.markdown(
        "A black rectangle after pressing START almost always means the media "
        "never arrived, rather than that detection failed.\n\n"
        "**Why.** WebRTC first tries to send video straight from your browser "
        "to the server. A STUN server only helps the two sides describe where "
        "they are; it cannot carry anything. Streamlit Community Cloud puts "
        "the app behind a proxy that refuses that direct path, so the "
        "connection negotiates successfully and then no frames flow.\n\n"
        "**The fix is a TURN server**, which relays the media instead. Free "
        "tiers exist — metered.ca's Open Relay gives 20 GB a month, Twilio's "
        "Network Traversal Service has trial credit. Sign up, then add the "
        "credentials to **Manage app → Settings → Secrets** on Streamlit "
        "Cloud:\n\n"
        "```toml\n"
        "[turn]\n"
        'urls = ["turn:standard.relay.metered.ca:80"]\n'
        'username = "your-username"\n'
        'credential = "your-credential"\n'
        "```\n\n"
        "The page picks them up on the next restart, and this warning "
        "disappears.\n\n"
        "**If you would rather not run a relay at all:** `python main.py` on "
        "your own machine is the same loop at full camera frame rate, with "
        "nothing to configure. The other two pages need no TURN server."
    )

st.caption(
    "Works today on the detector's COCO vocabulary: bottle, banana, apple, "
    "orange, book, cup, bowl, phone, laptop, scissors, cutlery, wine glass. "
    "Battery, aluminium can, cardboard and plastic bag are not classes this "
    "model was trained on — hold one up and watch the system say so instead "
    "of guessing."
)

with st.expander("Why this is slower than running it locally"):
    st.markdown(
        "Every frame is uploaded to the server, run through YOLO on a shared "
        "CPU (~100 ms), rendered, and sent back — so a few frames per second "
        "is the realistic ceiling here, and the belt advances in real time "
        "regardless of how many frames arrive.\n\n"
        "`python main.py` on your own machine keeps everything local and runs "
        "at camera frame rate. The hosted version exists so the loop can be "
        "*shown* to anyone with a link, not because it is the faster way to "
        "run it."
    )
