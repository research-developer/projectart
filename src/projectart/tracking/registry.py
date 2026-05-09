"""TrackedRegistry — the smart container.

Per frame:
1. Receive a fresh list of YOLO detections.
2. For each existing entity, find the best-matching detection by
   (class_id, IoU). Match if IoU >= IOU_THRESHOLD.
3. Matched entities → call `on_update`, transition to PRESENT.
4. Unmatched detections → instantiate the right `TrackedEntity` subclass
   (via `class_filter`) → fire `on_enter`.
5. Unmatched entities → if missed > LOST_AFTER_S, transition to LEAVING;
   if missed > GONE_AFTER_S, fire `on_leave` and drop.

The association is greedy IoU per class. For dense scenes (many
overlapping bboxes of the same class) we'd swap in a Hungarian matcher
or a real tracker (ByteTrack / SORT). The interface above doesn't change.
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
    ):
        # Map YOLO class_name → TrackedEntity subclass.
        self._by_class: dict[str, type[TrackedEntity]] = {}
        for ty in entity_types:
            for name in ty.class_filter():
                if name in self._by_class:
                    log.warning(
                        "duplicate entity_type for class %r: %s shadowed by %s",
                        name,
                        self._by_class[name].__name__,
                        ty.__name__,
                    )
                self._by_class[name] = ty
        self.entities: dict[int, TrackedEntity] = {}
        self.iou_threshold = iou_threshold
        self.fallback_type = fallback_type
        self._next_id = 1

    def consume(self, detections: list[Detection], ts: float | None = None) -> None:
        """One pass per video frame. Drives every transition in the
        entity lifecycle."""
        now = time.monotonic() if ts is None else float(ts)
        unmatched_dets = list(detections)

        # 1. Associate existing entities to detections by class + best IoU
        for entity in list(self.entities.values()):
            best = -1
            best_iou = self.iou_threshold
            for i, det in enumerate(unmatched_dets):
                if det.class_id != entity.class_id:
                    continue
                bbox = BBox.from_detection(det)
                iou = entity.last_bbox.iou(bbox)
                if iou >= best_iou:
                    best = i
                    best_iou = iou
            if best >= 0:
                det = unmatched_dets.pop(best)
                if entity.state in (EntityState.ENTERING, EntityState.LEAVING):
                    entity.state = EntityState.PRESENT
                entity.on_update(det, now)
            else:
                # No match this frame — promote to LEAVING / GONE based on age
                missing_for = now - entity.last_seen_ts
                if (
                    entity.state == EntityState.PRESENT
                    and missing_for >= entity.LOST_AFTER_S
                ):
                    entity.state = EntityState.LEAVING
                if missing_for >= entity.GONE_AFTER_S:
                    entity.state = EntityState.GONE

        # 2. Spawn entities for unmatched detections
        for det in unmatched_dets:
            ent_type = self._resolve_type(det)
            if ent_type is None:
                continue
            ent = ent_type(track_id=self._next_id, det=det, ts=now)
            self._next_id += 1
            self.entities[ent.track_id] = ent
            ent.on_enter()
            ent.state = EntityState.PRESENT

        # 3. Drop GONE entities (after firing on_leave)
        gone = [tid for tid, e in self.entities.items() if e.state == EntityState.GONE]
        for tid in gone:
            ent = self.entities.pop(tid)
            ent.on_leave()

    # ---- introspection helpers ----

    def of_class(self, class_name: str) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if e.class_name == class_name]

    def get(self, track_id: int) -> TrackedEntity | None:
        return self.entities.get(track_id)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities.values())

    # ---- internals ----

    def _resolve_type(self, det: Detection) -> type[TrackedEntity] | None:
        ty = self._by_class.get(det.class_name or "")
        if ty is not None:
            return ty
        if self.fallback_type is not None:
            return self.fallback_type
        return None
