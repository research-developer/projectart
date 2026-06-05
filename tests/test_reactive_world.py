import pytest

from projectart.reactive.behaviors import EntityView
from projectart.reactive.config import ReactiveConfig
from projectart.reactive.world import Simulator, TrackedView


def _cfg():
    return ReactiveConfig.from_dict({
        "objects": {
            "box":  {"radius": 0.06, "shape": "box", "kinematic": True, "color": "auto"},
            "ball": {"radius": 0.04, "shape": "circle", "kinematic": False, "color": "#39f"},
        },
        "spawn": [{"kind": "ball", "x": 0.5, "y": 0.5}],
        "rules": [{"match": {"class": "person"},
                   "action": {"spawn": "box", "behaviors": [{"follow": {"gain": 1.0}}]}}],
        "physics": {"friction": 0.0, "restitution": 1.0, "momentum_transfer": 1.0},
        "sim": {"tick_hz": 30}, "despawn_ms": 100,
    })


def _person(track_id, x, y, vx=0.0, vy=0.0):
    return TrackedView(track_id=track_id,
                       view=EntityView(x=x, y=y, vx=vx, vy=vy, class_name="person"))


def test_prespawn_ball_present():
    sim = Simulator(_cfg())
    snap = sim.tick([], dt=0.033)
    balls = [o for o in snap if o.kind == "ball"]
    assert len(balls) == 1


def test_person_spawns_one_box_that_follows_and_keeps_id():
    sim = Simulator(_cfg())
    sim.tick([_person(1, 0.2, 0.2)], dt=0.033)
    boxes = [o for o in sim.objects if o.kind == "box"]
    assert len(boxes) == 1
    box_id = boxes[0].id
    # follow gain 1.0 -> snaps to entity
    snap = sim.tick([_person(1, 0.8, 0.8)], dt=0.033)
    box = [o for o in snap if o.kind == "box"][0]
    assert box.id == box_id                     # SAME object (morph, not recreate)
    assert (box.x, box.y) == pytest.approx((0.8, 0.8))
    assert box.bound_track_id == 1


def test_departed_track_despawns_box():
    sim = Simulator(_cfg())
    sim.tick([_person(1, 0.5, 0.5)], dt=0.033)
    assert any(o.kind == "box" for o in sim.objects)
    # track gone; advance past despawn_ms
    sim.tick([], dt=0.2)
    sim.tick([], dt=0.2)
    assert not any(o.kind == "box" for o in sim.objects)


def test_reacquire_mid_despawn_cancels_fade():
    sim = Simulator(_cfg())
    sim.tick([_person(1, 0.5, 0.5)], dt=0.033)   # bind
    sim.tick([], dt=0.05)                          # partial despawn (despawn_ms=100)
    box = next(o for o in sim.objects if o.kind == "box")
    assert box.alpha < 1.0 and box.despawning
    sim.tick([_person(1, 0.5, 0.5)], dt=0.033)     # track returns
    box = next(o for o in sim.objects if o.kind == "box")
    assert box.alpha == pytest.approx(1.0)
    assert box.despawning is False


def test_fast_box_pushes_ball():
    sim = Simulator(_cfg())
    # Spawn box at the ball, moving fast in +x; ball should gain +x velocity.
    sim.tick([_person(1, 0.46, 0.5, vx=2.0)], dt=0.033)
    sim.tick([_person(1, 0.48, 0.5, vx=2.0)], dt=0.033)
    ball = [o for o in sim.objects if o.kind == "ball"][0]
    assert ball.vx > 0.0
