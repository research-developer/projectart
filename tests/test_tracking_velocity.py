# tests/test_tracking_velocity.py
from __future__ import annotations

import pytest

from projectart.detection.yolo_dots import Detection
from projectart.tracking.entity import TrackedEntity


def _det(cx, cy):
    return Detection(class_id=0, cx=cx, cy=cy, w=20, h=20, confidence=0.9, class_name="person")


def test_new_entity_zero_velocity_and_one_hit():
    e = TrackedEntity(track_id=1, det=_det(100, 100), ts=0.0)
    assert e.velocity == (0.0, 0.0)
    assert e.hits == 1
    assert e.confirmed is False
    assert e.center == (100.0, 100.0)


def test_constant_velocity_converges():
    # Move +10px/frame in x at 10 fps -> ~100 px/s. 1€ smoothing converges.
    e = TrackedEntity(track_id=1, det=_det(100, 100), ts=0.0)
    x = 100.0
    for i in range(1, 12):
        x += 10.0
        e.on_update(_det(x, 100), ts=i * 0.1)
    vx, vy = e.velocity
    assert vx == pytest.approx(100.0, rel=0.2)
    assert abs(vy) < 5.0
    assert e.hits == 12
