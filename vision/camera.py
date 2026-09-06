"""Frame source: a webcam or a video file, behind one interface."""

from __future__ import annotations

import logging

import cv2

from core.config import Config, camera_source

log = logging.getLogger(__name__)


class CameraUnavailable(RuntimeError):
    pass


class Camera:
    """Yields frames until the source ends or the caller stops.

    A video file can loop, which matters more than it sounds: a demo that
    depends on holding objects up to a webcam fails in a room with bad
    lighting, and a recorded clip always behaves the same way in a viva.
    """

    def __init__(self, cfg: Config):
        self.source = camera_source(cfg)
        self.is_file = isinstance(self.source, str)
        self.loop = bool(cfg.get("camera.loop_video", True)) and self.is_file
        self.width = int(cfg.get("camera.width", 960))
        self.height = int(cfg.get("camera.height", 540))
        self.frame_id = 0

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise CameraUnavailable(
                f"Could not open camera source {self.source!r}. "
                f"If this is a webcam index, check no other application holds "
                f"the camera; if it is a file, check the path and codec."
            )
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        log.info("Camera opened: %s (%s)", self.source,
                 "file" if self.is_file else "device")

    def read(self):
        """Returns a frame, or None when the source is exhausted."""
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        self.frame_id += 1
        if frame.shape[1] != self.width:
            scale = self.width / frame.shape[1]
            frame = cv2.resize(frame, (self.width, int(frame.shape[0] * scale)))
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *exc): self.release()
