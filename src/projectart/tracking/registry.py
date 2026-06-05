"""TrackedRegistry — the smart container.

Two association modes, chosen per frame:
  * If detections carry `track_id` (from a persistent tracker), associate by id
    (exact). This is the robust path — survives big bbox jumps / crossings.
  * Otherwise fall back to greedy IoU per class (legacy default).

Hysteresis:
  * `confirm_after_hits` — on_enter fires only once an entity has been seen this
    many frames (suppresses 1-frame false positives). Default 1 = immediate.
  * `lost_after_s` / `gone_after_s` — coast through brief dropouts before LEAVING
    / GONE. Default None = use the entity class attrs (0.5 / 2.0).
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

from ..detection.yolo_dots import Detection
from .entity import BBox, EntityState, TrackedEntity

log = logging.getLogger(__name__)

DEFAULT_IOU_THRESHOLD = 0.20


class TrackedRegistry:
    def __init__(
        self,
        entity_types: Iterable[type[TrackedEntity]],
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        fallback_type: type[TrackedEntity] | None = None,
        confirm_after_hits: int = 1,
        lost_after_s: float | None = None,
        gone_after_s: float | None = None,
        min_confidence: float = 0.0,
    ):
        self._by_class: dict[str, type[TrackedEntity]] = {}
        for ty in entity_types:
            for name in ty.class_filter():
                if name in self._by_class:
                    log.warning(
                        "duplicate entity_type for class %r: %s shadowed by %s",
                        name, self._by_class[name].__name__, ty.__name__,
                    )
                self._by_class[name] = ty
        self.entities: dict[int, TrackedEntity] = {}
        self.iou_threshold = iou_threshold
        self.fallback_type = fallback_type
        self.confirm_after_hits = max(1, int(confirm_after_hits))
        self.lost_after_s = lost_after_s
        self.gone_after_s = gone_after_s
        self.min_confidence = float(min_confidence)
        self._next_id = 1
        self._by_track: dict[int, int] = {}  # tracker id -> internal track_id

    def consume(self, detections: list[Detection], ts: float | None = None) -> None:
        now = time.monotonic() if ts is None else float(ts)
        dets = [d for d in detections if d.confidence >= self.min_confidence]
        use_track_ids = any(d.track_id is not None for d in dets)
        if use_track_ids:
            self._consume_by_track_id(dets, now)
        else:
            self._consume_by_iou(dets, now)
        self._age_and_confirm(now)
        self._drop_gone()

    # ---- association: tracker ids ----

    def _consume_by_track_id(self, dets: list[Detection], now: float) -> None:
        for det in dets:
            if det.track_id is None:
                continue
            internal = self._by_track.get(det.track_id)
            ent = self.entities.get(internal) if internal is not None else None
            if ent is not None:
                if ent.state in (EntityState.ENTERING, EntityState.LEAVING):
                    ent.state = EntityState.PRESENT
                ent.on_update(det, now)
            else:
                self._spawn(det, now)

    # ---- association: greedy IoU (legacy) ----

    def _consume_by_iou(self, dets: list[Detection], now: float) -> None:
        unmatched = list(dets)
        for entity in list(self.entities.values()):
            best, best_iou = -1, self.iou_threshold
            for i, det in enumerate(unmatched):
                if det.class_id != entity.class_id:
                    continue
                iou = entity.last_bbox.iou(BBox.from_detection(det))
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best >= 0:
                det = unmatched.pop(best)
                if entity.state in (EntityState.ENTERING, EntityState.LEAVING):
                    entity.state = EntityState.PRESENT
                entity.on_update(det, now)
        for det in unmatched:
            self._spawn(det, now)

    # ---- spawn / age / confirm / drop ----

    def _spawn(self, det: Detection, now: float) -> None:
        ent_type = self._resolve_type(det)
        if ent_type is None:
            return
        ent = ent_type(track_id=self._next_id, det=det, ts=now)
        self._next_id += 1
        self.entities[ent.track_id] = ent
        if det.track_id is not None:
            self._by_track[det.track_id] = ent.track_id

    def _age_and_confirm(self, now: float) -> None:
        for ent in self.entities.values():
            lost = self.lost_after_s if self.lost_after_s is not None else ent.LOST_AFTER_S
            gone = self.gone_after_s if self.gone_after_s is not None else ent.GONE_AFTER_S
            missing = now - ent.last_seen_ts
            if not ent.confirmed and ent.hits >= self.confirm_after_hits:
                ent.confirmed = True
                ent.state = EntityState.PRESENT
                ent.on_enter()
            if ent.state == EntityState.PRESENT and missing >= lost:
                ent.state = EntityState.LEAVING
            if missing >= gone:
                ent.state = EntityState.GONE

    def _drop_gone(self) -> None:
        gone = [tid for tid, e in self.entities.items() if e.state == EntityState.GONE]
        for tid in gone:
            ent = self.entities.pop(tid)
            if ent.track_key is not None:
                self._by_track.pop(ent.track_key, None)
            if ent.confirmed:
                ent.on_leave()

    # ---- introspection ----

    def of_class(self, class_name: str) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if e.class_name == class_name]

    def confirmed(self) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if e.confirmed]

    def get(self, track_id: int) -> TrackedEntity | None:
        return self.entities.get(track_id)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities.values())

    def _resolve_type(self, det: Detection) -> type[TrackedEntity] | None:
        ty = self._by_class.get(det.class_name or "")
        if ty is not None:
            return ty
        return self.fallback_type
