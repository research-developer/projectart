"""Built-in TrackedEntity subclasses for COCO classes we care about.

Each one demonstrates the lifecycle pattern: on_enter() fires a "hello"
side effect, on_update() is the per-frame keepalive, on_leave() cleans
up. Behaviors are wired via simple callbacks rather than subclassing
when the consumer wants to attach UI/audio without touching this file.

Side effects (sounds, screen overlays) are NOT done here — they're
emitted as `BehaviorEvent`s through the event bus. Renderer / audio
layers subscribe and react. Keeps tracking testable in isolation.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from .entity import Detection, TrackedEntity
from .events import BehaviorBus

log = logging.getLogger(__name__)


class Cat(TrackedEntity):
    """A cat. Fires `cat.appeared` on enter, `cat.left` on leave."""

    @classmethod
    def class_filter(cls) -> set[str]:
        return {"cat"}

    def __init__(self, *args, bus: BehaviorBus | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bus = bus

    def on_enter(self) -> None:
        log.info("Cat appeared (track_id=%d, conf=%.2f)", self.track_id, self.last_confidence)
        if self._bus:
            self._bus.emit("cat.appeared", entity=self)

    def on_update(self, det: Detection, ts: float) -> None:
        super().on_update(det, ts)
        if self._bus:
            self._bus.emit("cat.update", entity=self)

    def on_leave(self) -> None:
        log.info("Cat left (track_id=%d)", self.track_id)
        if self._bus:
            self._bus.emit("cat.left", entity=self)


class Person(TrackedEntity):
    """A person. Owns a `gloves` slot for finger/dot sub-trackers
    (populated later by the gloves input source)."""

    @classmethod
    def class_filter(cls) -> set[str]:
        return {"person"}

    def __init__(self, *args, bus: BehaviorBus | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bus = bus
        # Optional sub-tracker handle for glove dots associated with this person.
        # Populated by the gloves input source once it can attribute dots → person.
        self.attrs.setdefault("glove_dots", [])

    def on_enter(self) -> None:
        log.info("Person appeared (track_id=%d)", self.track_id)
        if self._bus:
            self._bus.emit("person.appeared", entity=self)

    def on_update(self, det: Detection, ts: float) -> None:
        super().on_update(det, ts)
        if self._bus:
            self._bus.emit("person.update", entity=self)

    def on_leave(self) -> None:
        log.info("Person left (track_id=%d)", self.track_id)
        if self._bus:
            self._bus.emit("person.left", entity=self)


class GenericEntity(TrackedEntity):
    """Catch-all subclass — used as `fallback_type` so unrecognised
    classes are still tracked (and visible in logs/HUD) without code."""

    @classmethod
    def class_filter(cls) -> set[str]:
        # Empty = the registry will only use this via fallback_type, never
        # by class match.
        return set()


def attach_callbacks(
    cls: type[TrackedEntity],
    *,
    on_enter: Callable[[TrackedEntity], None] | None = None,
    on_update: Callable[[TrackedEntity, Detection, float], None] | None = None,
    on_leave: Callable[[TrackedEntity], None] | None = None,
) -> type[TrackedEntity]:
    """Build a TrackedEntity subclass at runtime that wraps the given
    callbacks. Useful when you want to attach behavior without writing
    a class — e.g. quick prototyping a "dog → bark sound" reaction."""

    name = f"_Callback_{cls.__name__}"

    class _Wrapped(cls):  # type: ignore[misc, valid-type]
        def on_enter(self) -> None:
            super().on_enter()
            if on_enter:
                on_enter(self)

        def on_update(self, det, ts) -> None:
            super().on_update(det, ts)
            if on_update:
                on_update(self, det, ts)

        def on_leave(self) -> None:
            super().on_leave()
            if on_leave:
                on_leave(self)

    _Wrapped.__name__ = name
    _Wrapped.__qualname__ = name
    return _Wrapped
