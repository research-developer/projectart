# tests/test_tracking_hysteresis.py
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.tracking import TrackedRegistry
from projectart.tracking.builtins import Person


def _det(tid, cx=100.0, cy=100.0, conf=0.9):
    return Detection(class_id=0, cx=cx, cy=cy, w=20, h=20, confidence=conf,
                     class_name="person", track_id=tid)


def test_track_id_association_keeps_one_entity_when_bbox_jumps():
    reg = TrackedRegistry(entity_types=[Person])
    reg.consume([_det(7, cx=100)], ts=0.0)
    ent = next(iter(reg))
    # Large jump that greedy-IoU would treat as a new object — track_id keeps it.
    reg.consume([_det(7, cx=600)], ts=0.05)
    assert len(reg) == 1
    assert next(iter(reg)).track_id == ent.track_id


def test_confirm_after_hits_suppresses_one_frame_flicker():
    enters = []

    class WiredPerson(Person):
        def on_enter(self):
            enters.append(self.track_id)

    reg = TrackedRegistry(entity_types=[WiredPerson], confirm_after_hits=3)
    reg.consume([_det(1)], ts=0.0)      # hit 1 -> not confirmed
    assert enters == []
    assert next(iter(reg)).confirmed is False
    reg.consume([_det(1)], ts=0.03)     # hit 2
    assert enters == []
    reg.consume([_det(1)], ts=0.06)     # hit 3 -> confirmed -> on_enter
    assert enters == [next(iter(reg)).track_id]
    assert next(iter(reg)).confirmed is True


def test_default_confirm_is_immediate():
    enters = []

    class WiredPerson(Person):
        def on_enter(self):
            enters.append(self.track_id)

    reg = TrackedRegistry(entity_types=[WiredPerson])  # default confirm_after_hits=1
    reg.consume([_det(1)], ts=0.0)
    assert len(enters) == 1


def test_unconfirmed_entity_not_present_until_confirmed():
    from projectart.tracking import EntityState
    reg = TrackedRegistry(entity_types=[Person], confirm_after_hits=3)
    reg.consume([_det(1)], ts=0.0)        # hit 1
    e = next(iter(reg))
    assert e.confirmed is False and e.state == EntityState.ENTERING
    reg.consume([_det(1)], ts=0.03)       # hit 2 — still not confirmed
    assert e.confirmed is False and e.state == EntityState.ENTERING
    reg.consume([_det(1)], ts=0.06)       # hit 3 — confirmed -> PRESENT
    assert e.confirmed is True and e.state == EntityState.PRESENT
