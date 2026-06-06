"""Tests for the Trigger/Event layer over the TrackedRegistry."""
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.tracking import TrackedRegistry
from projectart.tracking.builtins import Cat, Person
from projectart.tracking.entity import BBox
from projectart.tracking.events import BehaviorBus
from projectart.tracking.triggers import (
    AppearTrigger,
    DisappearTrigger,
    IntersectTrigger,
    TriggerEngine,
    containment,
    size_bucket,
)

FRAME_AREA = 640 * 360  # 230400


def _det(class_id, name, cx=100.0, cy=100.0, w=40.0, h=40.0, conf=0.9):
    return Detection(class_id=class_id, cx=cx, cy=cy, w=w, h=h, confidence=conf, class_name=name)


def _cat(**kw):
    return _det(15, "cat", **kw)


def _person(**kw):
    return _det(0, "person", **kw)


def _reg():
    return TrackedRegistry(entity_types=[Cat, Person])


# ---- helpers -----

def test_size_bucket():
    assert size_bucket(0.2, 0.15) == "near"
    assert size_bucket(0.05, 0.15) == "far"


def test_containment_full_and_none():
    big = BBox(cx=100, cy=100, w=200, h=200)     # 0..200
    small = BBox(cx=100, cy=100, w=40, h=40)      # fully inside big
    assert containment(big, small) == 1.0          # smaller box fully covered
    far = BBox(cx=500, cy=500, w=40, h=40)
    assert containment(big, far) == 0.0


# ---- AppearTrigger -----

def test_appear_fires_once_with_far_size():
    reg, trig = _reg(), AppearTrigger("cat", near_area_frac=0.15)
    reg.consume([_cat(w=50, h=50)], ts=0.0)        # area 2500 -> far
    evs = trig.update(reg, ts=0.0, frame_area=FRAME_AREA)
    assert len(evs) == 1
    assert evs[0].kind == "appear" and evs[0].class_name == "cat" and evs[0].size == "far"
    # next frame, same cat -> no re-fire
    reg.consume([_cat(w=50, h=50)], ts=0.05)
    assert trig.update(reg, ts=0.05, frame_area=FRAME_AREA) == []


def test_appear_near_size_for_big_bbox():
    reg, trig = _reg(), AppearTrigger("cat", near_area_frac=0.15)
    reg.consume([_cat(w=220, h=220)], ts=0.0)      # area 48400 / 230400 = 0.21 -> near
    evs = trig.update(reg, ts=0.0, frame_area=FRAME_AREA)
    assert evs[0].size == "near"


def test_appear_refires_for_new_track_after_gone():
    reg, trig = _reg(), AppearTrigger("cat")
    reg.consume([_cat()], ts=0.0)
    assert len(trig.update(reg, 0.0, FRAME_AREA)) == 1
    # cat gone: advance past GONE_AFTER_S (2.0s) so the registry drops it
    reg.consume([], ts=3.0)
    assert trig.update(reg, 3.0, FRAME_AREA) == []
    # a fresh cat -> new track id -> appear fires again
    reg.consume([_cat()], ts=3.1)
    assert len(trig.update(reg, 3.1, FRAME_AREA)) == 1


# ---- DisappearTrigger -----

def test_disappear_fires_when_cat_dropped():
    reg, trig = _reg(), DisappearTrigger("cat")
    reg.consume([_cat()], ts=0.0)
    assert trig.update(reg, 0.0) == []             # present, nothing yet
    reg.consume([_cat()], ts=0.1)
    trig.update(reg, 0.1)
    # disappear: no detections, advance past gone timer -> registry drops it
    reg.consume([], ts=3.0)
    evs = trig.update(reg, 3.0)
    assert len(evs) == 1 and evs[0].kind == "disappear" and evs[0].class_name == "cat"
    # does not fire again
    reg.consume([], ts=3.1)
    assert trig.update(reg, 3.1) == []


# ---- IntersectTrigger -----

def _pid(tid, cx, cy, w, h):
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9,
                     class_name="person", track_id=tid)


def _cid(tid, cx, cy, w, h):
    return Detection(class_id=15, cx=cx, cy=cy, w=w, h=h, confidence=0.9,
                     class_name="cat", track_id=tid)


def test_intersect_then_separate():
    # Use explicit track ids (as a real tracker provides) so the cat ENTITY moves
    # out of the person box rather than spawning a new coasting track.
    reg = _reg()
    trig = IntersectTrigger("person", "cat", min_overlap=0.4)
    reg.consume([_pid(1, 100, 100, 200, 200), _cid(2, 100, 100, 40, 40)], ts=0.0)
    evs = trig.update(reg, 0.0)
    assert len(evs) == 1 and evs[0].kind == "intersect"
    assert evs[0].class_name == "person" and evs[0].other_class == "cat"
    # still overlapping -> no repeat
    reg.consume([_pid(1, 100, 100, 200, 200), _cid(2, 100, 100, 40, 40)], ts=0.1)
    assert trig.update(reg, 0.1) == []
    # same cat entity moves away -> separate
    reg.consume([_pid(1, 100, 100, 200, 200), _cid(2, 500, 500, 40, 40)], ts=0.2)
    evs = trig.update(reg, 0.2)
    assert len(evs) == 1 and evs[0].kind == "separate"


# ---- TriggerEngine + bus -----

def test_engine_emits_events_on_bus():
    bus = BehaviorBus()
    got = []
    bus.on("event.appear", lambda *, event: got.append(event))
    reg = _reg()
    engine = TriggerEngine([AppearTrigger("cat")], bus=bus)
    reg.consume([_cat()], ts=0.0)
    returned = engine.update(reg, ts=0.0, frame_area=FRAME_AREA)
    assert len(returned) == 1
    assert len(got) == 1 and got[0].kind == "appear"
