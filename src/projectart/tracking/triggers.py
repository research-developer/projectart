"""Triggers + Events: a thin rule layer over the TrackedRegistry.

A `Trigger` watches the registry each frame and emits `Event`s on transitions:
  * `AppearTrigger`   — a confirmed entity of a class shows up (with far/near size).
  * `DisappearTrigger`— a previously-seen entity of a class is gone.
  * `IntersectTrigger`— two classes' boxes start/stop "mostly" overlapping.

Tolerance is parameterized and generous by default: appearance/disappearance ride
the registry's own confirm/coast hysteresis (so brief flicker doesn't fire), and
intersection uses a containment ratio with a low threshold. Triggers only look at
`registry.confirmed()` entities.

`Event`s are plain data; the `TriggerEngine` collects them per frame and (optionally)
re-emits them on a `BehaviorBus` so audio / overlays / anything can react.
"""
from __future__ import annotations

from dataclasses import dataclass

from .entity import BBox


@dataclass(slots=True)
class Event:
    """One thing that happened. `size` is far/near (from bbox area fraction);
    `other_*`/`overlap` are populated for intersect/separate events."""

    kind: str                          # appear | disappear | intersect | separate
    class_name: str                    # primary subject, e.g. "cat"
    track_id: int
    size: str = "far"                  # far | near
    other_class: str | None = None     # the second class for intersect/separate
    other_track_id: int | None = None
    overlap: float = 0.0               # containment ratio for intersect/separate
    name: str | None = None            # recognized identity (recognize / named intersect)
    ts: float = 0.0


def size_bucket(area_fraction: float, near_area_frac: float) -> str:
    """far/near from a bbox's area as a fraction of the frame area."""
    return "near" if area_fraction >= near_area_frac else "far"


def containment(a: BBox, b: BBox) -> float:
    """Intersection area / smaller box area — "how much of the smaller box is
    inside the other". 1.0 = the smaller box is fully covered. Good proxy for
    "mostly intersecting"."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    smaller = min(a.w * a.h, b.w * b.h)
    return inter / smaller if smaller > 0 else 0.0


class AppearTrigger:
    """Fires once when a confirmed entity of `class_name` first appears."""

    def __init__(self, class_name: str, near_area_frac: float = 0.15):
        self.class_name = class_name
        self.near_area_frac = near_area_frac
        self._seen: set[int] = set()

    def update(self, registry, ts: float, frame_area: float = 0.0) -> list[Event]:
        events: list[Event] = []
        present: set[int] = set()
        for e in registry.confirmed():
            if e.class_name != self.class_name:
                continue
            present.add(e.track_id)
            if e.track_id not in self._seen:
                area = e.last_bbox.w * e.last_bbox.h
                frac = area / frame_area if frame_area > 0 else 0.0
                events.append(Event(
                    kind="appear", class_name=self.class_name, track_id=e.track_id,
                    size=size_bucket(frac, self.near_area_frac), ts=ts,
                ))
        self._seen = present  # forget gone ids so a fresh track re-triggers
        return events


class DisappearTrigger:
    """Fires once when a previously-confirmed entity of `class_name` is gone
    (the registry drops it after its generous gone-timer)."""

    def __init__(self, class_name: str):
        self.class_name = class_name
        self._known: set[int] = set()

    def update(self, registry, ts: float, frame_area: float = 0.0) -> list[Event]:
        present = {e.track_id for e in registry.confirmed() if e.class_name == self.class_name}
        gone = self._known - present
        self._known = present | (self._known & present)
        return [Event(kind="disappear", class_name=self.class_name, track_id=tid, ts=ts)
                for tid in sorted(gone)]


class IntersectTrigger:
    """Fires `intersect` when an `class_a` box and a `class_b` box start overlapping
    by at least `min_overlap` (containment ratio), and `separate` when they stop."""

    def __init__(self, class_a: str, class_b: str, min_overlap: float = 0.4):
        self.class_a = class_a
        self.class_b = class_b
        self.min_overlap = min_overlap
        self._pairs: set[tuple[int, int]] = set()

    def update(self, registry, ts: float, frame_area: float = 0.0) -> list[Event]:
        a_ents = {e.track_id: e for e in registry.confirmed() if e.class_name == self.class_a}
        b_ents = {e.track_id: e for e in registry.confirmed() if e.class_name == self.class_b}
        current: dict[tuple[int, int], float] = {}
        for aid, a in a_ents.items():
            for bid, b in b_ents.items():
                ov = containment(a.last_bbox, b.last_bbox)
                if ov >= self.min_overlap:
                    current[(aid, bid)] = ov
        events: list[Event] = []
        for pair in sorted(current.keys() - self._pairs):
            a = a_ents.get(pair[0])
            name = a.attrs.get("name") if a is not None else None
            events.append(Event(kind="intersect", class_name=self.class_a, track_id=pair[0],
                                other_class=self.class_b, other_track_id=pair[1],
                                overlap=current[pair], name=name, ts=ts))
        for pair in sorted(self._pairs - current.keys()):
            events.append(Event(kind="separate", class_name=self.class_a, track_id=pair[0],
                                other_class=self.class_b, other_track_id=pair[1], ts=ts))
        self._pairs = set(current.keys())
        return events


class RecognizeTrigger:
    """Fires `recognize` once when a confirmed entity of `class_name` first carries
    a name (set externally, e.g. by face recognition into ``entity.attrs['name']``)."""

    def __init__(self, class_name: str = "person"):
        self.class_name = class_name
        self._announced: dict[int, str] = {}  # track_id -> announced name

    def update(self, registry, ts: float, frame_area: float = 0.0) -> list[Event]:
        events: list[Event] = []
        present: set[int] = set()
        for e in registry.confirmed():
            if e.class_name != self.class_name:
                continue
            present.add(e.track_id)
            name = e.attrs.get("name")
            if name and self._announced.get(e.track_id) != name:
                self._announced[e.track_id] = name
                events.append(Event(kind="recognize", class_name=self.class_name,
                                    track_id=e.track_id, name=name, ts=ts))
        for tid in [t for t in self._announced if t not in present]:
            del self._announced[tid]
        return events


class TriggerEngine:
    """Runs a list of triggers each frame; optionally re-emits Events on a bus."""

    def __init__(self, triggers: list, bus=None):
        self.triggers = list(triggers)
        self.bus = bus

    def update(self, registry, ts: float, frame_area: float = 0.0) -> list[Event]:
        events: list[Event] = []
        for trig in self.triggers:
            events.extend(trig.update(registry, ts, frame_area))
        if self.bus is not None:
            for ev in events:
                self.bus.emit("event", event=ev)
                self.bus.emit(f"event.{ev.kind}", event=ev)
        return events
