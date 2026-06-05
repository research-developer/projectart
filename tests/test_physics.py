
import pytest

from projectart.reactive.config import PhysicsConfig
from projectart.reactive.objects import ReactiveObject


def _ball(x, y, vx=0.0, vy=0.0, r=0.05, m=1.0):
    return ReactiveObject(id=1, kind="ball", x=x, y=y, vx=vx, vy=vy, radius=r, mass=m)


def test_integrate_moves_and_applies_friction():
    from projectart.reactive.physics import integrate

    p = PhysicsConfig(friction=1.0, gravity=(0.0, 0.0))
    o = _ball(0.0, 0.0, vx=1.0)
    integrate(o, dt=0.1, physics=p)
    assert o.x == pytest.approx(0.1)
    assert o.vx == pytest.approx(0.9)  # 1.0 * (1 - 1.0*0.1)


def test_kinematic_not_integrated():
    from projectart.reactive.physics import integrate

    o = ReactiveObject(id=1, kind="box", x=0.5, y=0.5, vx=5.0, kinematic=True)
    integrate(o, dt=0.1, physics=PhysicsConfig())
    assert (o.x, o.vx) == (0.5, 5.0)


def test_kinematic_imparts_momentum_to_ball():
    from projectart.reactive.physics import resolve_collisions

    p = PhysicsConfig(restitution=1.0, momentum_transfer=1.0)
    hand = ReactiveObject(id=1, kind="box", x=0.50, y=0.5, vx=1.0, radius=0.05, kinematic=True)
    ball = ReactiveObject(id=2, kind="ball", x=0.58, y=0.5, vx=0.0, radius=0.05, mass=1.0)
    resolve_collisions([hand, ball], p)
    assert ball.vx > 0.0  # pushed in +x (swipe direction)
    assert hand.vx == 1.0  # kinematic unaffected


def test_dynamic_pair_conserves_momentum():
    from projectart.reactive.physics import resolve_collisions

    p = PhysicsConfig(restitution=1.0, momentum_transfer=1.0)
    a = ReactiveObject(id=1, kind="ball", x=0.50, y=0.5, vx=1.0, radius=0.05, mass=1.0)
    b = ReactiveObject(id=2, kind="ball", x=0.58, y=0.5, vx=0.0, radius=0.05, mass=1.0)
    p0 = a.mass * a.vx + b.mass * b.vx
    resolve_collisions([a, b], p)
    p1 = a.mass * a.vx + b.mass * b.vx
    assert p1 == pytest.approx(p0, abs=1e-9)


def test_no_collision_when_apart():
    from projectart.reactive.physics import resolve_collisions

    a = _ball(0.1, 0.1, vx=0.0)
    b = _ball(0.9, 0.9, vx=0.0)
    resolve_collisions([a, b], PhysicsConfig())
    assert a.vx == 0.0 and b.vx == 0.0
