"""Scene input source — watches the Yi cameras with YOLO and maintains a
TrackedRegistry of cats / people / arbitrary objects.

Pipeline per frame (cam-A; cam-B is optional and currently used only for
ping-pong-fallback if cam-A's queue is empty — stereo here would be
overkill for scene tagging):

    cam-A frame → YOLO detect → registry.consume(detections)
                                    │
                                    ├── entity hooks (on_enter, on_update, on_leave)
                                    │       fire BehaviorBus events for in-process
                                    │       reactions (audio, side effects)
                                    │
                                    └── delta diff vs last frame
                                            broadcast EntityEvent over WS so the
                                            renderer can fade overlays in/out.

This is the smallest piece that makes "see cat → react on screen" actually
happen. Behaviors are wired via `BehaviorBus` and are entirely the consumer's
problem (renderer subscribes to wire events; audio plugs into the bus).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..audio.cat_audio import CatAudioConfig, CatAudioPlayer
from ..capture.yi_rtsp import YiCapture, yi_rtsp_url
from ..detection.yolo_dots import DotDetector
from ..server.protocol import EntityEvent
from ..server.ws import Server
from ..tracking import TrackedEntity, TrackedRegistry
from ..tracking.builtins import Cat, GenericEntity, Person
from ..tracking.events import BehaviorBus
from ..tracking.triggers import (
    AppearTrigger,
    DisappearTrigger,
    IntersectTrigger,
    RecognizeTrigger,
    TriggerEngine,
)

log = logging.getLogger(__name__)


def _identity_canvas_mapper(canvas_size: tuple[int, int]):
    """Linear cam-pixel → canvas-pixel rescale. Wrong for projector
    geometry; right enough to show overlays for cats/people."""
    cw, ch = canvas_size

    def to_canvas(
        cam_xywh: tuple[float, float, float, float], frame_shape
    ) -> tuple[float, float, float, float]:
        h, w = frame_shape[:2]
        cx, cy, bw, bh = cam_xywh
        sx, sy = cw / max(1, w), ch / max(1, h)
        return (cx - bw / 2) * sx, (cy - bh / 2) * sy, bw * sx, bh * sy

    return to_canvas


def _assign_face_names(registry, results) -> None:
    """Set ``entity.attrs['name']`` on the person whose box contains each recognized
    face's centre. `results` = [((x, y, w, h), name, score), ...]."""
    persons = [e for e in registry if e.class_name == "person"]
    for (fx, fy, fw, fh), name, _score in results:
        if not name:
            continue
        cx, cy = fx + fw / 2, fy + fh / 2
        for e in persons:
            b = e.last_bbox
            if b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2:
                e.attrs["name"] = name
                break


class ScenePublisher:
    """Owns the camera capture, the YOLO detector, the registry, and the
    BehaviorBus. One per cam-pair; small enough to compose into a multi-cam
    setup later."""

    def __init__(
        self,
        canvas_size: tuple[int, int],
        server: Server,
        camera_url_a: str,
        camera_url_b: str | None = None,
        yolo_weights_path: str | None = None,
        target_hz: int = 15,
        entity_types: list[type[TrackedEntity]] | None = None,
        fallback_type: type[TrackedEntity] | None = GenericEntity,
        bus: BehaviorBus | None = None,
        enable_cat_audio: bool = True,
        audio_device: str | None = None,
        confirm_after_hits: int = 2,
        gone_after_s: float = 2.5,
        near_area_frac: float = 0.18,
        intersect_overlap: float = 0.4,
        enable_face_recognition: bool = True,
        face_gallery_path: str | None = None,
        recognize_every_s: float = 0.5,
    ):
        self.canvas_size = canvas_size
        self.server = server
        self.target_hz = target_hz
        self._period = 1.0 / max(1, target_hz)

        self.capture_a = YiCapture(url=camera_url_a, name="cam-a")
        self.capture_b: YiCapture | None = (
            YiCapture(url=camera_url_b, name="cam-b") if camera_url_b else None
        )
        self.detector = DotDetector(weights_path=yolo_weights_path)

        self.bus = bus if bus is not None else BehaviorBus()
        # Generous, parameterized tolerance: confirm over a couple frames (kills
        # one-frame false cats) and coast through brief dropouts before "gone".
        self.registry = TrackedRegistry(
            entity_types=entity_types if entity_types is not None else [Cat, Person],
            fallback_type=fallback_type,
            confirm_after_hits=confirm_after_hits,
            gone_after_s=gone_after_s,
        )
        # Triggers -> Events on the bus (appear/disappear cat, person∩cat intersect).
        self.triggers = TriggerEngine(
            [
                AppearTrigger("cat", near_area_frac=near_area_frac),
                DisappearTrigger("cat"),
                IntersectTrigger("person", "cat", min_overlap=intersect_overlap),
                RecognizeTrigger("person"),
            ],
            bus=self.bus,
        )
        self.cat_audio: CatAudioPlayer | None = None
        if enable_cat_audio:
            self.cat_audio = CatAudioPlayer(CatAudioConfig(device=audio_device))
            self.cat_audio.subscribe(self.bus)

        # Face recognition (optional): names person entities -> RecognizeTrigger /
        # name-aware intersect. Off if no gallery exists.
        self._face_rec = None
        self._gallery = None
        self._recognize_every_s = recognize_every_s
        self._last_recognize = 0.0
        if enable_face_recognition:
            gp = (Path(face_gallery_path).expanduser() if face_gallery_path
                  else Path("~/.projectart/faces/gallery.npz").expanduser())
            if gp.exists():
                try:
                    from ..detection.faces import FaceGallery, FaceRecognizer
                    self._gallery = FaceGallery.load(gp)
                    self._face_rec = FaceRecognizer()
                    log.info("face recognition on (%d enrolled)", len(self._gallery.names()))
                except Exception:
                    log.exception("face recognition disabled (load failed)")
            else:
                log.info("no face gallery at %s; face recognition off", gp)

        self._to_canvas = _identity_canvas_mapper(canvas_size)
        # delta-diff state for emitting enter/update/leave wire events
        self._known_bboxes: dict[int, tuple[float, float, float, float]] = {}
        self._known_meta: dict[int, tuple[str, float]] = {}     # track_id → (class_name, last_conf)

    # ---- public lifecycle ----

    async def run(self) -> None:
        log.info(
            "scene publisher starting (cam-a=%s, cam-b=%s, target_hz=%d)",
            self.capture_a.url,
            self.capture_b.url if self.capture_b else "(none)",
            self.target_hz,
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

    # ---- helpers used by tests / external callers ----

    def step(self, detections: list, ts: float) -> list[EntityEvent]:
        """Pure: take a list of `Detection` objects + a timestamp, run them
        through the registry, and return the list of EntityEvents the
        publisher would broadcast. No camera, no I/O. Used by tests and
        callers that pre-detected on their own pipeline."""
        self.registry.consume(detections, ts=ts)
        return self._compute_events(ts_ms=int(ts * 1000), frame_shape=(1080, 1920, 3))

    # ---- inner loop ----

    async def _loop(self) -> None:
        while True:
            t_iter_start = time.monotonic()
            frame = self.capture_a.latest()
            if frame is None and self.capture_b is not None:
                frame = self.capture_b.latest()
            if frame is None:
                await asyncio.sleep(self._period)
                continue

            detections = self.detector(frame.image)
            now = time.monotonic()
            self.registry.consume(detections, ts=now)

            # Throttled face recognition -> name person entities (before triggers,
            # so RecognizeTrigger / named intersect see fresh names).
            recog_due = (now - self._last_recognize) >= self._recognize_every_s
            if self._face_rec is not None and recog_due:
                self._last_recognize = now
                try:
                    _assign_face_names(
                        self.registry, self._face_rec.identify(frame.image, self._gallery))
                except Exception:
                    log.exception("face recognition step failed")

            # Evaluate triggers -> Events on the bus (cat audio reacts via subscription).
            frame_area = float(frame.image.shape[0] * frame.image.shape[1])
            self.triggers.update(self.registry, now, frame_area)

            events = self._compute_events(ts_ms=frame.ts_ms, frame_shape=frame.image.shape)
            for ev in events:
                await self.server.broadcast(ev)
                # Also fire on the in-process bus so audio / overlays /
                # any other Python-side reactions can subscribe.
                self.bus.emit(f"entity.{ev.phase}", event=ev)
                self.bus.emit(f"entity.{ev.phase}.{ev.class_name}", event=ev)

            elapsed = time.monotonic() - t_iter_start
            await asyncio.sleep(max(0.0, self._period - elapsed))

    # ---- delta-diff (broken out for testability) ----

    def _compute_events(self, ts_ms: int, frame_shape) -> list[EntityEvent]:
        events: list[EntityEvent] = []
        current_ids: set[int] = set()

        for ent in self.registry:
            current_ids.add(ent.track_id)
            bbox = self._to_canvas(
                (ent.last_bbox.cx, ent.last_bbox.cy, ent.last_bbox.w, ent.last_bbox.h),
                frame_shape,
            )
            if ent.track_id in self._known_bboxes:
                phase = "update"
            else:
                phase = "enter"
            events.append(
                EntityEvent(
                    track_id=ent.track_id,
                    class_name=ent.class_name,
                    phase=phase,
                    bbox_x=bbox[0], bbox_y=bbox[1],
                    bbox_w=bbox[2], bbox_h=bbox[3],
                    confidence=ent.last_confidence,
                    ts_ms=ts_ms,
                )
            )
            self._known_bboxes[ent.track_id] = bbox
            self._known_meta[ent.track_id] = (ent.class_name, ent.last_confidence)

        # Departed = previously known but not in current registry.
        departed = set(self._known_bboxes.keys()) - current_ids
        for tid in departed:
            bbox = self._known_bboxes.pop(tid)
            class_name, conf = self._known_meta.pop(tid, ("unknown", 0.0))
            events.append(
                EntityEvent(
                    track_id=tid,
                    class_name=class_name,
                    phase="leave",
                    bbox_x=bbox[0], bbox_y=bbox[1],
                    bbox_w=bbox[2], bbox_h=bbox[3],
                    confidence=conf,
                    ts_ms=ts_ms,
                )
            )

        return events


def build_scene_source(
    canvas_size: tuple[int, int],
    server: Server,
    webcam_a: str | None,
    webcam_b: str | None,
    yolo_weights: str | None,
    target_hz: int = 15,
    enable_cat_audio: bool = True,
    audio_device: str | None = None,
    enable_face_recognition: bool = True,
) -> ScenePublisher:
    """CLI bridge — same yi-hack-v5 URL defaults as the gloves source."""
    url_a = webcam_a or yi_rtsp_url(host="10.0.0.33", low_res=True)
    url_b = webcam_b or yi_rtsp_url(host="10.0.0.34", low_res=True)
    return ScenePublisher(
        canvas_size=canvas_size,
        server=server,
        camera_url_a=url_a,
        camera_url_b=url_b,
        yolo_weights_path=yolo_weights,
        target_hz=target_hz,
        enable_cat_audio=enable_cat_audio,
        audio_device=audio_device,
        enable_face_recognition=enable_face_recognition,
    )
