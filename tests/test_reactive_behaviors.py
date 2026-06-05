import pytest

from projectart.reactive.behaviors import EntityView, apply_behavior
from projectart.reactive.objects import ReactiveObject


def _view(x=0.5, y=0.5, vx=0.0, vy=0.0, bbox_area=0.04, conf=0.9, dwell=0.0, cls="person"):
    return EntityView(x=x, y=y, vx=vx, vy=vy, bbox_area=bbox_area, confidence=conf,
                      dwell_s=dwell, class_name=cls)


def test_follow_eases_toward_entity():
    o = ReactiveObject(id=1, kind="box", x=0.0, y=0.0, kinematic=True)
    apply_behavior(o, ("follow", {"gain": 0.5}), _view(x=1.0, y=1.0))
    assert o.x == pytest.approx(0.5)
    assert o.y == pytest.approx(0.5)


def test_follow_negative_gain_flees():
    o = ReactiveObject(id=1, kind="box", x=0.5, y=0.5, kinematic=True)
    apply_behavior(o, ("follow", {"gain": -0.5}), _view(x=0.5, y=0.5))
    # at same point, fleeing does nothing; offset entity to verify direction
    o2 = ReactiveObject(id=2, kind="box", x=0.4, y=0.5, kinematic=True)
    apply_behavior(o2, ("follow", {"gain": -0.5}), _view(x=0.5, y=0.5))
    assert o2.x < 0.4  # moves away from entity


def test_scale_from_bbox_clamped():
    o = ReactiveObject(id=1, kind="box", x=0, y=0)
    apply_behavior(o, ("scale", {"source": "bbox", "min": 0.03, "max": 0.12}), _view(bbox_area=1.0))
    assert o.radius == pytest.approx(0.12)
    apply_behavior(o, ("scale", {"source": "bbox", "min": 0.03, "max": 0.12}), _view(bbox_area=0.0))
    assert o.radius == pytest.approx(0.03)


def test_colorize_speed_to_hue_sets_color_and_state():
    o = ReactiveObject(id=1, kind="box", x=0, y=0)
    apply_behavior(o, ("colorize", {"source": "velocity", "mapping": "speed_to_hue"}),
                   _view(vx=2.0, vy=0.0))
    assert o.color.startswith("hsl(")
    o_slow = ReactiveObject(id=2, kind="box", x=0, y=0)
    apply_behavior(o_slow, ("colorize", {"source": "velocity", "mapping": "speed_to_hue"}),
                   _view(vx=0.0, vy=0.0))
    assert o_slow.color != o.color


def test_unknown_behavior_is_noop():
    o = ReactiveObject(id=1, kind="box", x=0.5, y=0.5)
    apply_behavior(o, ("nonsense", {}), _view())
    assert (o.x, o.y) == (0.5, 0.5)
