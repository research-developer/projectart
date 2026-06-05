"""Reactive input source: capture -> tracker -> registry -> simulator -> SceneFrame.

`step()` is pure (no camera) for tests. `run()` adds the live loop. World
coordinates come from CamToWorld; world velocity is a finite-difference of
mapped centres kept per track (correct for identity or homography mapping).
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..capture.yi_rtsp import YiCapture, yi_rtsp_url
from ..detection.yolo_dots import Detection, DotDetector
from ..geometry.mapping import CamToWorld
from ..reactive.behaviors import EntityView
from ..reactive.config import ReactiveConfig, load_default
from ..reactive.world import Simulator, TrackedView
from ..server.protocol import SceneFrame, SceneObject
from ..server.ws import Server
from ..tracking import TrackedRegistry
from ..tracking.builtins import GenericEntity, Person

log = logging.getLogger(__name__)


class ReactiveSource:
    def __init__(
        self,
        canvas_size: tuple[int, int],
        server: Server | None,
        config: ReactiveConfig,
        camera_url_a: str | None = None,
        yolo_weights_path: str | None = None,
        frame_size: tuple[int, int] = (640, 360),
    ):
        self.canvas_size = canvas_size
        self.server = server
        self.config = config
        self.frame_w, self.frame_h = frame_size
        self.camera_url_a = camera_url_a
        self._period = 1.0 / max(1, config.sim.tick_hz)

        self.cam_to_world = CamToWorld.from_config(config.world.cam_to_world)
        self.registry = TrackedRegistry(
            entity_types=[Person],
            fallback_type=GenericEntity,
            confirm_after_hits=config.tracking.confirm_after_hits,
            lost_after_s=config.tracking.lost_after_s,
            gone_after_s=config.tracking.gone_after_s,
            min_confidence=config.tracking.min_confidence,
        )
        self.sim = Simulator(config)
        self._prev_world: dict[int, tuple[float, float, float]] = {}  # track_id -> (x,y,ts)

        self.capture_a: YiCapture | None = None
        self.detector: DotDetector | None = None
        if camera_url_a is not None:
            self.capture_a = YiCapture(url=camera_url_a, name="cam-a")
            self.detector = DotDetector(weights_path=yolo_weights_path)

    @classmethod
    def for_testing(cls, config: ReactiveConfig, frame_w: int, frame_h: int) -> ReactiveSource:
        return cls(canvas_size=(1920, 1080), server=None, config=config,
                   camera_url_a=None, frame_size=(frame_w, frame_h))

    # ---- pure step (no camera) ----

    def step(self, detections: list[Detection], ts: float) -> SceneFrame:
        self.registry.consume(detections, ts=ts)
        views: list[TrackedView] = []
        for ent in self.registry.confirmed():
            wx, wy = self.cam_to_world(ent.center[0], ent.center[1], self.frame_w, self.frame_h)
            tkey = ent.track_key if ent.track_key is not None else ent.track_id
            vx = vy = 0.0
            prev = self._prev_world.get(tkey)
            if prev is not None:
                pdt = max(1e-3, ts - prev[2])
                vx, vy = (wx - prev[0]) / pdt, (wy - prev[1]) / pdt
            self._prev_world[tkey] = (wx, wy, ts)
            frame_area = float(self.frame_w * self.frame_h)
            bbox_area = (ent.last_bbox.w * ent.last_bbox.h) / max(1.0, frame_area)
            views.append(TrackedView(
                track_id=tkey,
                view=EntityView(x=wx, y=wy, vx=vx, vy=vy, bbox_area=bbox_area,
                                confidence=ent.last_confidence,
                                dwell_s=ts - ent.first_seen_ts, class_name=ent.class_name),
            ))
        objs = self.sim.tick(views, dt=self._period)
        return SceneFrame(
            ts_ms=int(ts * 1000),
            objects=[SceneObject(
                id=o.id, kind=o.kind, shape=o.shape, x=o.x, y=o.y, vx=o.vx, vy=o.vy,
                r=o.radius, color=o.color, state=o.state, alpha=o.alpha, angle=o.angle,
                track_id=o.bound_track_id) for o in objs],
        )

    # ---- live loop ----

    async def run(self) -> None:
        assert self.capture_a is not None and self.detector is not None and self.server is not None
        log.info("reactive source starting (cam-a=%s, tick_hz=%d)",
                 self.capture_a.url, self.config.sim.tick_hz)
        self.capture_a.start()
        loop_t0 = time.monotonic()
        try:
            while True:
                t0 = time.monotonic()
                frame = self.capture_a.latest()
                if frame is None:
                    await asyncio.sleep(self._period)
                    continue
                self.frame_h, self.frame_w = frame.image.shape[:2]
                dets = self.detector.track(frame.image, tracker=self.config.tracking.tracker)
                scene = self.step(dets, ts=time.monotonic() - loop_t0)
                await self.server.broadcast(scene)
                await asyncio.sleep(max(0.0, self._period - (time.monotonic() - t0)))
        finally:
            self.capture_a.stop()


def build_reactive_source(
    canvas_size: tuple[int, int],
    server: Server,
    webcam_a: str | None,
    yolo_weights: str | None,
    config_path: str | None = None,
) -> ReactiveSource:
    config = ReactiveConfig.from_json(config_path) if config_path else load_default()
    url_a = webcam_a or yi_rtsp_url(host="10.0.0.33", low_res=True)
    return ReactiveSource(canvas_size=canvas_size, server=server, config=config,
                          camera_url_a=url_a, yolo_weights_path=yolo_weights)
