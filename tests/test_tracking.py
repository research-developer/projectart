"""End-to-end tests for the tracking module: lifecycle transitions,
registry routing, behavior bus, IoU association."""
from __future__ import annotations

import time

import pytest

from projectart.detection.yolo_dots import Detection
from projectart.tracking import EntityState, TrackedEntity, TrackedRegistry
from projectart.tracking.builtins import Cat, GenericEntity, Person, attach_callbacks
from projectart.tracking.entity import BBox
from projectart.tracking.events import BehaviorBus


def _det(class_id: int, name: str, cx=100.0, cy=100.0, w=20.0, h=20.0, conf=0.9):
    return Detection(class_id=class_id, cx=cx, cy=cy, w=w, h=h, confidence=conf, class_name=name)


# ---- BBox -----


def test_bbox_iou_disjoint_is_zero():
    a = BBox(cx=10, cy=10, w=20, h=20)   # 0..20
    b = BBox(cx=100, cy=100, w=20, h=20) # 90..110
    assert a.iou(b) == 0.0


def test_bbox_iou_self_is_one():
    a = BBox(cx=10, cy=10, w=20, h=20)
    assert a.iou(a) == pytest.approx(1.0)


def test_bbox_iou_partial():
    a = BBox(cx=0, cy=0, w=10, h=10)     # -5..5 in both axes
    b = BBox(cx=5, cy=0, w=10, h=10)     # 0..10 / -5..5
    # intersection = 5*10 = 50; union = 100 + 100 - 50 = 150 → 1/3
    assert a.iou(b) == pytest.approx(1.0 / 3.0)


# ---- Registry routing -----


def test_registry_spawns_correct_subclass():
    reg = TrackedRegistry(entity_types=[Cat, Person])
    reg.consume([_det(15, "cat"), _det(0, "person")], ts=0.0)
    assert len(reg) == 2
    classes = sorted(type(e).__name__ for e in reg)
    assert classes == ["Cat", "Person"]


def test_registry_unknown_class_dropped_without_fallback():
    reg = TrackedRegistry(entity_types=[Cat])
    reg.consume([_det(99, "giraffe")], ts=0.0)
    assert len(reg) == 0


def test_registry_unknown_class_uses_fallback():
    reg = TrackedRegistry(entity_types=[Cat], fallback_type=GenericEntity)
    reg.consume([_det(99, "giraffe")], ts=0.0)
    assert len(reg) == 1
    assert isinstance(next(iter(reg)), GenericEntity)


def test_registry_associates_same_track_across_frames():
    reg = TrackedRegistry(entity_types=[Cat])
    reg.consume([_det(15, "cat", cx=100, cy=100)], ts=0.0)
    cat = next(iter(reg))
    track_id = cat.track_id

    # next frame, slight shift — should associate, NOT spawn a new entity
    reg.consume([_det(15, "cat", cx=105, cy=100)], ts=0.05)
    assert len(reg) == 1
    assert next(iter(reg)).track_id == track_id


def test_registry_low_iou_spawns_new_track():
    reg = TrackedRegistry(entity_types=[Cat], iou_threshold=0.5)
    reg.consume([_det(15, "cat", cx=100, cy=100, w=20, h=20)], ts=0.0)
    # Box is far away → IoU=0 → no association → new track
    reg.consume([_det(15, "cat", cx=500, cy=500, w=20, h=20)], ts=0.05)
    assert len(reg) == 2


# ---- Lifecycle transitions -----


def test_entity_enters_present_then_leaves():
    bus = BehaviorBus()
    seen: dict[str, int] = {"appeared": 0, "left": 0}
    bus.on("cat.appeared", lambda *, entity: seen.__setitem__("appeared", seen["appeared"] + 1))
    bus.on("cat.left", lambda *, entity: seen.__setitem__("left", seen["left"] + 1))

    # Bind the bus to the Cat class via a small subclass:
    class WiredCat(Cat):
        def __init__(self, **kwargs):
            super().__init__(bus=bus, **kwargs)

    reg = TrackedRegistry(entity_types=[WiredCat])

    reg.consume([_det(15, "cat", cx=100, cy=100)], ts=0.0)
    assert seen == {"appeared": 1, "left": 0}
    cat = next(iter(reg))
    assert cat.state == EntityState.PRESENT  # promoted from ENTERING after on_enter

    # Frame at +0.1s, cat still seen
    reg.consume([_det(15, "cat", cx=101, cy=101)], ts=0.1)
    assert cat.state == EntityState.PRESENT

    # Skip several frames — past LOST_AFTER_S (0.5s) and GONE_AFTER_S (2.0s)
    reg.consume([], ts=1.0)   # 0.9s missed → LEAVING
    assert cat.state == EntityState.LEAVING
    reg.consume([], ts=3.0)   # 2.9s missed → GONE → fires on_leave + drops
    assert len(reg) == 0
    assert seen == {"appeared": 1, "left": 1}


def test_re_acquire_after_brief_loss():
    """A leaving entity can return to PRESENT if it reappears within
    GONE_AFTER_S."""
    reg = TrackedRegistry(entity_types=[Cat])
    reg.consume([_det(15, "cat")], ts=0.0)
    cat = next(iter(reg))

    reg.consume([], ts=0.7)   # > LOST_AFTER_S → LEAVING
    assert cat.state == EntityState.LEAVING

    reg.consume([_det(15, "cat")], ts=0.9)   # back, before GONE
    assert cat.state == EntityState.PRESENT
    assert len(reg) == 1


def test_attach_callbacks_factory():
    counts = {"enter": 0, "leave": 0}
    Bouncy = attach_callbacks(
        Cat,
        on_enter=lambda e: counts.__setitem__("enter", counts["enter"] + 1),
        on_leave=lambda e: counts.__setitem__("leave", counts["leave"] + 1),
    )
    reg = TrackedRegistry(entity_types=[Bouncy])
    reg.consume([_det(15, "cat")], ts=0.0)
    reg.consume([], ts=10.0)
    assert counts == {"enter": 1, "leave": 1}


# ---- BehaviorBus -----


def test_bus_handler_exception_doesnt_stop_others():
    bus = BehaviorBus()
    seen = []
    bus.on("x", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.on("x", lambda **kw: seen.append(kw["a"]))
    bus.emit("x", a=42)
    assert seen == [42]


def test_bus_off_removes_handler():
    bus = BehaviorBus()
    seen = []
    h = lambda **kw: seen.append(kw)  # noqa: E731
    bus.on("x", h)
    bus.off("x", h)
    bus.emit("x", a=1)
    assert seen == []


def test_bus_emit_forwards_event_kwarg():
    """Regression: scene.py calls `bus.emit("entity.enter", event=ev)`. The
    topic parameter must not be named `event`, or that kwarg collides and
    raises TypeError. Surfaced the first time the live scene loop ran."""
    bus = BehaviorBus()
    got = []
    bus.on("entity.enter", lambda **kw: got.append(kw))
    bus.emit("entity.enter", event="ENTITY_OBJ")
    assert got == [{"event": "ENTITY_OBJ"}]
