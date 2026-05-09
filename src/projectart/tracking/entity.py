"""TrackedEntity — long-lived state for one tracked object across frames.

Lifecycle:

    [no entity]
        │  first detection of class C with no nearby existing entity
        ▼
    ENTERING  ──► on_enter() fires once
        │  next frame still seeing it
        ▼
    PRESENT   ──► on_update(det) fires every frame it's seen
        │  no detection for `lost_after_s` seconds
        ▼
    LEAVING   (still tracked, can re-acquire)
        │  no detection for `gone_after_s` seconds
        ▼
    GONE      ──► on_leave() fires once, registry drops the entity

Subclasses override the three hooks. `class_filter()` declares which YOLO
class names this subclass handles — the registry uses it to pick the right
constructor for a fresh detection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

# Re-exported for convenience so callers can import everything from .entity
from ..detection.yolo_dots import Detection  # noqa: F401


@dataclass(slots=True)
class BBox:
    """Axis-aligned box in pixel space (cam pixels for raw YOLO output;
    canvas pixels once mapped through the homography)."""

    cx: float
    cy: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter == 0.0:
            return 0.0
        a = self.w * self.h
        b = other.w * other.h
        union = a + b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def from_detection(cls, det: Detection) -> "BBox":
        return cls(cx=det.cx, cy=det.cy, w=det.w, h=det.h)


class EntityState(Enum):
    ENTERING = "entering"
    PRESENT = "present"
    LEAVING = "leaving"
    GONE = "gone"


class TrackedEntity:
    """Base class for any tracked object. Subclass and override the
    lifecycle hooks (`on_enter`, `on_update`, `on_leave`) to attach
    rule-based behavior. Overriding `class_filter` is required — the
    registry uses it to pick the right subclass for fresh detections.

    Subclasses should NOT mutate `state` directly — the registry owns
    state transitions. Subclasses can store their own attributes (e.g.
    `Cat` might own a screen-overlay handle, `Person` might own a list
    of currently-detected glove dots).
    """

    # Class-level: tuning knobs (override per subclass when useful).
    LOST_AFTER_S: float = 0.5     # frames missing → LEAVING
    GONE_AFTER_S: float = 2.0     # frames missing → GONE

    def __init__(self, track_id: int, det: Detection, ts: float | None = None):
        self.track_id = track_id
        self.class_name = det.class_name or f"class_{det.class_id}"
        self.class_id = det.class_id
        self.state = EntityState.ENTERING
        self.last_bbox = BBox.from_detection(det)
        self.last_confidence = float(det.confidence)
        now = time.monotonic() if ts is None else float(ts)
        self.first_seen_ts = now
        self.last_seen_ts = now
        # Each subclass can attach its own per-instance state; this dict
        # is a free-form bag for behaviors that don't want to subclass.
        self.attrs: dict = {}

    # ---- lifecycle hooks (subclasses override) ------------------------------

    def on_enter(self) -> None:
        """Called exactly once, the first frame this entity appears."""

    def on_update(self, det: Detection, ts: float) -> None:
        """Called every frame this entity is detected (after the first).
        Default updates the bbox + last-seen timestamp; subclasses can
        extend (call `super().on_update(det, ts)` first)."""
        self.last_bbox = BBox.from_detection(det)
        self.last_confidence = float(det.confidence)
        self.last_seen_ts = ts

    def on_leave(self) -> None:
        """Called exactly once when the entity transitions to GONE.
        Use this to free resources (overlay handles, sounds, etc.)."""

    # ---- class-method registry helpers --------------------------------------

    @classmethod
    def class_filter(cls) -> set[str]:
        """YOLO class names this entity type accepts. Empty = wildcard
        (rare; used by `GenericEntity` for opt-in catch-all tracking)."""
        return set()

    # ---- representation -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.track_id}, "
            f"class={self.class_name!r}, state={self.state.value})"
        )
