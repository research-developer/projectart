"""Glove input source — single-camera M2 path.

Pipeline per frame:
    cam (Q=1 RTSP) → YOLO dot detection → choose primary dot → 1€ filter
       → homography (cam pixels → canvas pixels) → PointerEvent + HudAnchorEvent

For M2 the homography comes from the 4-corner glove poke wizard (not yet
implemented — `wizard_4corner.run()` will block startup once it lands).
For now, if no calibration is loaded, we use an identity-ish scaling that
maps the camera frame directly to the canvas. That's wrong but it lets us
exercise the full pipeline and see SOMETHING on the canvas during dev.

Stereo + wall-plane contact gating is M3 (`stereo.py`). In M2, "contact"
is a simple "any dot detected" heuristic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from ..capture.yi_rtsp import YiCapture, yi_rtsp_url
from ..detection.yolo_dots import Detection, DotDetector
from ..geometry.filter_one_euro import OneEuroFilter
from ..server.protocol import HudAnchorEvent, PointerEvent
from ..server.ws import Server

log = logging.getLogger(__name__)


def _identity_mapper(canvas_size: tuple[int, int]):
    """Fallback when no calibration is available: rescale camera-pixel
    detections to canvas size. Wrong, but makes the pipeline visible."""
    cw, ch = canvas_size

    def map_xy(cam_x: float, cam_y: float, frame_shape: tuple[int, int, int]) -> tuple[float, float]:
        h, w = frame_shape[:2]
        return (cam_x / max(1, w)) * cw, (cam_y / max(1, h)) * ch

    return map_xy


def _pick_primary_dot(detections: list[Detection]) -> Optional[Detection]:
    """Choose the highest-confidence detection as the primary tip.

    Once the YOLO classes are confirmed (one per finger?), this should
    prefer the index-tip class. For now: argmax confidence."""
    if not detections:
        return None
    return max(detections, key=lambda d: d.confidence)


class GlovesSource:
    """Single-camera glove tracking. Runs the inference loop in an asyncio task
    that periodically yields control so the websocket server can drain."""

    def __init__(
        self,
        canvas_size: tuple[int, int],
        server: Server,
        camera_url: str,
        yolo_weights_path: Optional[str] = None,
        target_hz: int = 30,
    ):
        self.canvas_size = canvas_size
        self.server = server
        self.target_hz = target_hz
        self._period = 1.0 / max(1, target_hz)

        self.capture = YiCapture(url=camera_url, name="cam-a")
        self.detector = DotDetector(weights_path=yolo_weights_path)
        self.tip_filter = OneEuroFilter(mincutoff=1.0, beta=0.05)
        self.hud_filter = OneEuroFilter(mincutoff=0.7, beta=0.10)
        self._mapper = _identity_mapper(canvas_size)

    async def run(self) -> None:
        log.info("gloves input starting (camera=%s)", self.capture.url)
        self.capture.start()
        try:
            await self._loop()
        finally:
            self.capture.stop()

    async def _loop(self) -> None:
        loop_t0 = time.monotonic()
        last_pos: Optional[tuple[float, float]] = None
        while True:
            t_iter_start = time.monotonic()
            frame = self.capture.latest()
            if frame is None:
                await asyncio.sleep(self._period)
                continue

            # YOLO inference — sync; on the asyncio loop. For M2 it's fine
            # because the target is 30 Hz and ultralytics CPU inference is
            # ~30-60 ms on M-series. We'll punt to a thread pool if profiling
            # shows the loop blocking the websocket.
            detections = self.detector(frame.image)
            primary = _pick_primary_dot(detections)
            now_s = time.monotonic() - loop_t0
            ts_ms = frame.ts_ms

            if primary is not None:
                cam_x, cam_y = primary.cx, primary.cy
                cx, cy = self._mapper(cam_x, cam_y, frame.image.shape)
                smoothed = self.tip_filter(np.array([cx, cy], dtype=np.float32), t=now_s)
                sx, sy = float(smoothed[0]), float(smoothed[1])
                velocity = 0.0
                if last_pos is not None:
                    dx = sx - last_pos[0]
                    dy = sy - last_pos[1]
                    velocity = (dx * dx + dy * dy) ** 0.5
                last_pos = (sx, sy)

                pe = PointerEvent(
                    x=sx,
                    y=sy,
                    contact=True,           # M2 placeholder; M3 swaps in stereo gating
                    velocity=velocity,
                    confidence=primary.confidence,
                    ts_ms=ts_ms,
                )
                await self.server.broadcast(pe)

                # HUD anchor — median of all detected dots, separately filtered
                if detections:
                    anchor_x = float(np.median([d.cx for d in detections]))
                    anchor_y = float(np.median([d.cy for d in detections]))
                    ax, ay = self._mapper(anchor_x, anchor_y, frame.image.shape)
                    hud_smoothed = self.hud_filter(np.array([ax, ay], dtype=np.float32), t=now_s)
                    he = HudAnchorEvent(
                        x=float(hud_smoothed[0]),
                        y=float(hud_smoothed[1]),
                        visible=True,
                        ts_ms=ts_ms,
                    )
                    await self.server.broadcast(he)
            else:
                # No detections this frame — emit a "lift" with contact=False
                # so the renderer ends the current stroke.
                if last_pos is not None:
                    pe = PointerEvent(
                        x=last_pos[0], y=last_pos[1],
                        contact=False,
                        velocity=0.0,
                        ts_ms=ts_ms,
                    )
                    await self.server.broadcast(pe)
                    last_pos = None

            # Pace the loop to target_hz
            elapsed = time.monotonic() - t_iter_start
            sleep_for = max(0.0, self._period - elapsed)
            await asyncio.sleep(sleep_for)


def build_gloves_source(
    canvas_size: tuple[int, int],
    server: Server,
    webcam_a: Optional[str],
    yolo_weights: Optional[str],
) -> GlovesSource:
    """Helper that resolves a camera URL from CLI args or falls back to the
    default yi-hack-v5 URL on 10.0.0.33."""
    url = webcam_a or yi_rtsp_url(host="10.0.0.33", low_res=True)
    return GlovesSource(
        canvas_size=canvas_size,
        server=server,
        camera_url=url,
        yolo_weights_path=yolo_weights,
    )
