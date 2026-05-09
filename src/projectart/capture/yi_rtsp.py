"""Low-latency RTSP capture for the yi-hack-v5 cameras.

Each camera runs in its own daemon thread. The thread continuously reads
the newest frame and stores it in a 1-slot queue, dropping anything older.
Inference loops pull "newest only" — they never block on capture and they
never see stale frames.

FFmpeg latency hygiene (per spec §2):
    OPENCV_FFMPEG_CAPTURE_OPTIONS=fflags;nobuffer|flags;low_delay|tune;zerolatency
    cv2.CAP_PROP_BUFFERSIZE = 1

yi-hack-v5 exposes:
    rtsp://<ip>/ch0_0.h264      high-res H.264
    rtsp://<ip>/ch0_1.h264      low-res H.264 (preferred for inference)
    rtsp://<ip>/ch0_2.h264      audio-only
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LOW_RES_PATH = "ch0_1.h264"
DEFAULT_HIGH_RES_PATH = "ch0_0.h264"

# Set once for the whole process. Safe to set multiple times.
_FFMPEG_OPTS = "fflags;nobuffer|flags;low_delay|tune;zerolatency"
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_OPTS)


def yi_rtsp_url(host: str, low_res: bool = True, user: str = "", password: str = "") -> str:
    """Compose the standard yi-hack-v5 RTSP URL for a host."""
    auth = ""
    if user:
        auth = f"{user}:{password}@" if password else f"{user}@"
    path = DEFAULT_LOW_RES_PATH if low_res else DEFAULT_HIGH_RES_PATH
    return f"rtsp://{auth}{host}/{path}"


@dataclass(slots=True)
class Frame:
    image: np.ndarray   # HxWxC, BGR (OpenCV default)
    ts_ms: int


class YiCapture:
    """One-frame queue capture from a single RTSP source.

    `latest()` returns the most recent frame or None if no frame has arrived
    yet. The queue is depth-1 with newest-wins semantics; stale frames are
    dropped at write time, never at read time.
    """

    def __init__(self, url: str, name: str = "", reconnect_delay_s: float = 2.0):
        self.url = url
        self.name = name or url.rsplit("/", 1)[-1]
        self.reconnect_delay_s = reconnect_delay_s
        self._q: Queue[Frame] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"yi[{self.name}]", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self, timeout_s: float = 0.0) -> Optional[Frame]:
        if timeout_s <= 0.0:
            try:
                return self._q.get_nowait()
            except Empty:
                return None
        try:
            return self._q.get(timeout=timeout_s)
        except Empty:
            return None

    # -------- internals --------

    def _put_newest(self, frame: Frame) -> None:
        # Drop any stale frame so the queue holds the newest exactly.
        try:
            self._q.get_nowait()
        except Empty:
            pass
        try:
            self._q.put_nowait(frame)
        except Exception:
            pass

    def _loop(self) -> None:
        # Lazy import keeps cv2 out of the API surface.
        import cv2

        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                log.warning("[%s] failed to open %s; retry in %.1fs", self.name, self.url, self.reconnect_delay_s)
                cap.release()
                if self._stop.wait(self.reconnect_delay_s):
                    return
                continue

            log.info("[%s] capture open: %s", self.name, self.url)
            consecutive_fails = 0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok or img is None:
                    consecutive_fails += 1
                    if consecutive_fails > 30:
                        log.warning("[%s] too many read failures; reconnecting", self.name)
                        break
                    time.sleep(0.01)
                    continue
                consecutive_fails = 0
                self._put_newest(Frame(image=img, ts_ms=int(time.monotonic() * 1000)))

            cap.release()
            if not self._stop.is_set():
                log.info("[%s] reconnecting in %.1fs", self.name, self.reconnect_delay_s)
                if self._stop.wait(self.reconnect_delay_s):
                    return
