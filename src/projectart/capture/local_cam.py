"""Low-latency local webcam capture (Continuity Camera / USB).

Mirrors YiCapture: one daemon thread, depth-1 newest-wins queue, `latest()`
returns the most recent frame or None. AVFoundation backend on macOS. cv2 is
imported lazily.

IMPORTANT (Continuity Camera): disable auto-effects, ESPECIALLY Center Stage —
its per-frame reframing breaks the fixed camera->stage homography. Use a short
shutter with enough light to avoid motion blur on fast swings.
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue

from .yi_rtsp import Frame  # reuse the (image, ts_ms) dataclass

log = logging.getLogger(__name__)


class LocalCamera:
    def __init__(self, index: int = 0, name: str = "local", width: int = 1280, height: int = 720):
        self.index = index
        self.name = name
        self.width = width
        self.height = height
        self._q: Queue[Frame] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"localcam[{self.name}]", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self) -> Frame | None:
        try:
            return self._q.get_nowait()
        except Empty:
            return None

    def _put_newest(self, frame: Frame) -> None:
        try:
            self._q.get_nowait()
        except Empty:
            pass
        try:
            self._q.put_nowait(frame)
        except Exception:
            pass

    def _loop(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            pass
        if not cap.isOpened():
            log.error("[%s] could not open camera index %d", self.name, self.index)
            cap.release()
            return
        log.info("[%s] local camera open (index=%d)", self.name, self.index)
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok or img is None:
                time.sleep(0.005)
                continue
            self._put_newest(Frame(image=img, ts_ms=int(time.monotonic() * 1000)))
        cap.release()
