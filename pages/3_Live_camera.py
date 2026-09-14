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

# Google's public STUN server, which is what lets the browser and the server
# find each other through NAT. Without it the handshake never completes on
# most home networks.
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


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

controls = st.columns([1, 1, 1.3])
with controls[0]:
    view = st.radio("View", ["Control room", "Camera only"],
                    help="Control room is the full desktop view: camera, "
                         "decision readout, belt and bins. Camera only is "
                         "just the annotated video, which is easier to read "
                         "on a phone.")
with controls[1]:
    size = st.select_slider(
        "Detection size", [320, 416, 640], value=416,
        help="Smaller is faster and less accurate. 640 is what the live "
             "system uses; 320 keeps a slow connection watchable.")
with controls[2]:
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


def make_callback(plant: Plant, control_room: bool):
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
            detections = plant.pipe.process_frame(img, plant.frame_id,
                                                  img.shape[1])
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


ctx = webrtc_streamer(
    key="waste-line",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=make_callback(plant, view == "Control room"),
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
