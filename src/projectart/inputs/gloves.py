"""Glove input source — single-camera (M2) or stereo (M3) pipelines.

Single-camera pipeline (M2):
    cam (Q=1 RTSP) → YOLO → primary dot → 1€ filter
       → identity / 4-corner homography → PointerEvent + HudAnchorEvent

Stereo pipeline (M3):
    cam-A (Q=1)        cam-B (Q=1)
        \\\\               //
         YOLO            YOLO
            \\\\         //
            correspond_by_class
                  │
              triangulate (StereoRig)
                  │
            wall plane signed distance  →  contact gating
                  │
              project to canvas         →  PointerEvent + HudAnchorEvent

The stereo pipeline activates when:
  * a second camera URL is provided AND
  * a calibration file is loaded that contains stereo extrinsics + wall plane

Otherwise we fall back to single-cam with the M2 placeholder homography.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..calibration.persist import CalibrationDoc, load_calibration
from ..capture.yi_rtsp import YiCapture, yi_rtsp_url
from ..detection.stereo import StereoRig, correspond_by_class, triangulate_pair
from ..detection.yolo_dots import Detection, DotDetector
from ..geometry.filter_one_euro import OneEuroFilter
from ..geometry.wall_plane import (
    Plane,
    PlaneBasis,
    project_to_uv,
    signed_distance_to_plane,
)
from ..server.protocol import HudAnchorEvent, PointerEvent
from ..server.ws import Server

log = logging.getLogger(__name__)


def _identity_mapper(canvas_size: tuple[int, int]):
    """Fallback when no calibration is available: rescale camera-pixel
    detections to canvas size. Wrong, but visible — useful in dev."""
    cw, ch = canvas_size

    def map_xy(
        cam_x: float, cam_y: float, frame_shape: tuple[int, int, int]
    ) -> tuple[float, float]:
        h, w = frame_shape[:2]
        return (cam_x / max(1, w)) * cw, (cam_y / max(1, h)) * ch

    return map_xy


def _pick_primary_dot(detections: list[Detection]) -> Detection | None:
    """Highest-confidence detection. Once we have per-finger classes, this
    should prefer index-tip; for now confidence is fine."""
    if not detections:
        return None
    return max(detections, key=lambda d: d.confidence)


def _stereo_rig_from_calib(calib: CalibrationDoc) -> StereoRig | None:
    """Build a StereoRig from a CalibrationDoc, or None if the doc lacks
    the required fields."""
    if calib.camera_a is None or calib.camera_b is None or calib.stereo is None:
        return None
    K_a = np.asarray(calib.camera_a.K, dtype=np.float64)
    K_b = np.asarray(calib.camera_b.K, dtype=np.float64)
    R = np.asarray(calib.stereo.R_a_to_b, dtype=np.float64)
    t = np.asarray(calib.stereo.t_a_to_b, dtype=np.float64)
    dist_a = np.asarray(calib.camera_a.dist, dtype=np.float64)
    dist_b = np.asarray(calib.camera_b.dist, dtype=np.float64)
    return StereoRig(K_a=K_a, K_b=K_b, R_a_to_b=R, t_a_to_b=t, dist_a=dist_a, dist_b=dist_b)


def _wall_plane_from_calib(calib: CalibrationDoc) -> tuple[Plane, PlaneBasis | None] | None:
    """Build a wall Plane (and optional PlaneBasis) from a CalibrationDoc."""
    if calib.wall_plane is None:
        return None
    plane = Plane(
        normal=np.asarray(calib.wall_plane.normal, dtype=np.float64),
        centroid=np.asarray(calib.wall_plane.centroid, dtype=np.float64),
    )
    basis: PlaneBasis | None = None
    if calib.uv_basis is not None:
        basis = PlaneBasis(
            plane=plane,
            u=np.asarray(calib.uv_basis.u, dtype=np.float64),
            v=np.asarray(calib.uv_basis.v, dtype=np.float64),
        )
    return plane, basis


class GlovesSource:
    """Glove tracking. Runs an asyncio loop that pulls frames, runs YOLO,
    optionally triangulates with the second camera, and broadcasts pointer
    + HUD events."""

    def __init__(
        self,
        canvas_size: tuple[int, int],
        server: Server,
        camera_url_a: str,
        camera_url_b: str | None = None,
        yolo_weights_path: str | None = None,
        calibration: CalibrationDoc | None = None,
        target_hz: int = 30,
    ):
        self.canvas_size = canvas_size
        self.server = server
        self.target_hz = target_hz
        self._period = 1.0 / max(1, target_hz)

        self.capture_a = YiCapture(url=camera_url_a, name="cam-a")
        self.capture_b: YiCapture | None = (
            YiCapture(url=camera_url_b, name="cam-b") if camera_url_b else None
        )
        self.detector_a = DotDetector(weights_path=yolo_weights_path)
        self.detector_b = (
            DotDetector(weights_path=yolo_weights_path) if camera_url_b else None
        )
        self.tip_filter = OneEuroFilter(mincutoff=1.0, beta=0.05)
        self.hud_filter = OneEuroFilter(mincutoff=0.7, beta=0.10)
        self._mapper = _identity_mapper(canvas_size)

        # Stereo + wall plane (only active when calibration provides both).
        self.calibration = calibration
        self._rig: StereoRig | None = None
        self._wall: Plane | None = None
        self._uv_basis: PlaneBasis | None = None
        self._contact_eps_m: float = 0.015
        self._stereo_active = False

        if calibration is not None:
            self._rig = _stereo_rig_from_calib(calibration)
            wp = _wall_plane_from_calib(calibration)
            if wp is not None:
                self._wall, self._uv_basis = wp
            self._contact_eps_m = float(calibration.contact_epsilon_m)

        if self.capture_b is not None and self._rig is not None and self._wall is not None:
            self._stereo_active = True
            log.info("stereo pipeline active (rig + wall plane loaded)")
        elif self.capture_b is not None:
            log.warning(
                "stereo capture configured but calibration missing rig and/or wall plane; "
                "falling back to cam-A only path"
            )

    async def run(self) -> None:
        log.info(
            "gloves input starting (cam-a=%s, cam-b=%s, stereo=%s)",
            self.capture_a.url,
            self.capture_b.url if self.capture_b else "(none)",
            self._stereo_active,
        )
        self.capture_a.start()
        if self.capture_b is not None:
            self.capture_b.start()
        try:
            await self._loop()
        finally:
            self.capture_a.stop()
            if self.capture_b is not None:
                self.capture_b.stop()

    async def _loop(self) -> None:
        loop_t0 = time.monotonic()
        last_pos: tuple[float, float] | None = None
        while True:
            t_iter_start = time.monotonic()
            frame_a = self.capture_a.latest()
            if frame_a is None:
                await asyncio.sleep(self._period)
                continue

            now_s = time.monotonic() - loop_t0
            ts_ms = frame_a.ts_ms

            detections_a = self.detector_a(frame_a.image)

            if self._stereo_active:
                pe, he = self._stereo_step(frame_a, detections_a, now_s, ts_ms)
            else:
                pe, he = self._mono_step(frame_a, detections_a, now_s, ts_ms, last_pos)

            if pe is not None:
                if pe.contact and (pe.x or pe.y):
                    last_pos = (pe.x, pe.y)
                elif not pe.contact:
                    last_pos = None
                await self.server.broadcast(pe)
            if he is not None:
                await self.server.broadcast(he)

            elapsed = time.monotonic() - t_iter_start
            await asyncio.sleep(max(0.0, self._period - elapsed))

    # ---- monocular step (M2) ----

    def _mono_step(
        self,
        frame_a,
        detections_a: list[Detection],
        now_s: float,
        ts_ms: int,
        last_pos: tuple[float, float] | None,
    ) -> tuple[PointerEvent | None, HudAnchorEvent | None]:
        primary = _pick_primary_dot(detections_a)
        if primary is None:
            if last_pos is not None:
                return (
                    PointerEvent(
                        x=last_pos[0], y=last_pos[1], contact=False,
                        velocity=0.0, ts_ms=ts_ms,
                    ),
                    None,
                )
            return None, None

        cam_x, cam_y = primary.cx, primary.cy
        cx, cy = self._mapper(cam_x, cam_y, frame_a.image.shape)
        smoothed = self.tip_filter(np.array([cx, cy], dtype=np.float32), t=now_s)
        sx, sy = float(smoothed[0]), float(smoothed[1])
        velocity = 0.0
        if last_pos is not None:
            dx = sx - last_pos[0]
            dy = sy - last_pos[1]
            velocity = (dx * dx + dy * dy) ** 0.5
        pe = PointerEvent(
            x=sx, y=sy, contact=True,
            velocity=velocity,
            confidence=primary.confidence,
            ts_ms=ts_ms,
        )
        he = self._hud_event(detections_a, frame_a.image.shape, now_s, ts_ms)
        return pe, he

    # ---- stereo step (M3) ----

    def _stereo_step(
        self,
        frame_a,
        detections_a: list[Detection],
        now_s: float,
        ts_ms: int,
    ) -> tuple[PointerEvent | None, HudAnchorEvent | None]:
        assert self.capture_b is not None
        assert self.detector_b is not None
        assert self._rig is not None
        assert self._wall is not None

        frame_b = self.capture_b.latest()
        if frame_b is None:
            # B not ready yet — degrade gracefully to cam-A only this tick.
            return self._mono_step(frame_a, detections_a, now_s, ts_ms, None)
        detections_b = self.detector_b(frame_b.image)

        pairs = correspond_by_class(detections_a, detections_b)
        if not pairs:
            return None, None

        # Pick the highest-confidence pair as the primary tip.
        a, b = max(pairs, key=lambda p: min(p[0].confidence, p[1].confidence))
        try:
            point_3d = triangulate_pair(self._rig, (a.cx, a.cy), (b.cx, b.cy))
        except Exception:
            log.exception("triangulation failed for class %d", a.class_id)
            return None, None
        if not np.all(np.isfinite(point_3d)):
            return None, None

        # Contact gating against wall plane.
        signed = signed_distance_to_plane(point_3d, self._wall)
        contact = abs(signed) < self._contact_eps_m

        # Project the 3D point onto the wall plane → in-plane (u, v) → canvas.
        if self._uv_basis is not None:
            uv = project_to_uv(point_3d, self._uv_basis)
            cx, cy = self._uv_to_canvas(uv)
        else:
            # No basis — fall back to cam-A homography placeholder
            cx, cy = self._mapper(a.cx, a.cy, frame_a.image.shape)

        smoothed = self.tip_filter(np.array([cx, cy], dtype=np.float32), t=now_s)
        sx, sy = float(smoothed[0]), float(smoothed[1])

        pe = PointerEvent(
            x=sx, y=sy,
            contact=bool(contact),
            velocity=0.0,                     # 1€ filter could expose this; punt for now
            confidence=min(a.confidence, b.confidence),
            ts_ms=ts_ms,
        )
        he = self._hud_event(detections_a, frame_a.image.shape, now_s, ts_ms)
        return pe, he

    # ---- helpers ----

    def _hud_event(
        self,
        detections: list[Detection],
        frame_shape,
        now_s: float,
        ts_ms: int,
    ) -> HudAnchorEvent | None:
        if not detections:
            return None
        anchor_x = float(np.median([d.cx for d in detections]))
        anchor_y = float(np.median([d.cy for d in detections]))
        ax, ay = self._mapper(anchor_x, anchor_y, frame_shape)
        hud_smoothed = self.hud_filter(np.array([ax, ay], dtype=np.float32), t=now_s)
        return HudAnchorEvent(
            x=float(hud_smoothed[0]),
            y=float(hud_smoothed[1]),
            visible=True,
            ts_ms=ts_ms,
        )

    def _uv_to_canvas(self, uv: np.ndarray) -> tuple[float, float]:
        """If the calibration includes an in-plane → canvas homography we
        could apply it here; for M3 we treat (u, v) as already-meter-scaled
        and rescale linearly to the canvas. The wizards (M5) will fix this."""
        if (
            self.calibration is not None
            and self.calibration.homography_uv_to_canvas is not None
        ):
            H = np.asarray(self.calibration.homography_uv_to_canvas, dtype=np.float64)
            p = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
            q = H @ p
            if abs(q[2]) > 1e-9:
                return float(q[0] / q[2]), float(q[1] / q[2])
        # No homography → simple linear rescale assuming a ~1.5 m wall span.
        cw, ch = self.canvas_size
        # Reasonable default: u in [-0.75, 0.75] m → 0..cw
        scale_x = cw / 1.5
        scale_y = ch / 1.0
        return (uv[0] + 0.75) * scale_x, (uv[1] + 0.5) * scale_y


def build_gloves_source(
    canvas_size: tuple[int, int],
    server: Server,
    webcam_a: str | None,
    webcam_b: str | None,
    yolo_weights: str | None,
    calibration: CalibrationDoc | None = None,
) -> GlovesSource:
    """CLI bridge — fills in the standard yi-hack-v5 URLs for the two
    in-house cameras when not overridden, and tries to load calibration
    from the default path if not passed explicitly."""
    url_a = webcam_a or yi_rtsp_url(host="10.0.0.33", low_res=True)
    url_b = webcam_b or yi_rtsp_url(host="10.0.0.34", low_res=True)
    calib = calibration if calibration is not None else load_calibration()
    return GlovesSource(
        canvas_size=canvas_size,
        server=server,
        camera_url_a=url_a,
        camera_url_b=url_b,
        yolo_weights_path=yolo_weights,
        calibration=calib,
    )
