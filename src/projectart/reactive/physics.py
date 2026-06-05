"""2D physics for reactive objects (world units, [0,1]).

Dynamic objects integrate with velocity + friction + gravity. Circle-circle
collisions resolve with positional separation and an impulse using restitution
and a tunable momentum_transfer scale. Kinematic objects (track-driven) have
inv_mass 0: they impart momentum but are not pushed — the "swipe -> ball flies"
mechanic.
"""
from __future__ import annotations

import math

from .config import PhysicsConfig
from .objects import ReactiveObject


def integrate(obj: ReactiveObject, dt: float, physics: PhysicsConfig) -> None:
    if obj.kinematic:
        return
    gx, gy = physics.gravity
    obj.vx += gx * dt
    obj.vy += gy * dt
    obj.x += obj.vx * dt
    obj.y += obj.vy * dt
    damp = max(0.0, 1.0 - physics.friction * dt)
    obj.vx *= damp
    obj.vy *= damp


def _resolve_pair(a: ReactiveObject, b: ReactiveObject, physics: PhysicsConfig) -> None:
    dx = b.x - a.x
    dy = b.y - a.y
    dist = math.hypot(dx, dy)
    min_dist = a.radius + b.radius
    if dist >= min_dist:
        return
    inv_a, inv_b = a.inv_mass, b.inv_mass
    inv_sum = inv_a + inv_b
    if inv_sum == 0.0:
        return  # both kinematic — nothing to push
    if dist < 1e-9:
        nx, ny, dist = 1.0, 0.0, 1e-9
    else:
        nx, ny = dx / dist, dy / dist
    # Positional separation (split by inverse mass).
    overlap = min_dist - dist
    a.x -= nx * overlap * (inv_a / inv_sum)
    a.y -= ny * overlap * (inv_a / inv_sum)
    b.x += nx * overlap * (inv_b / inv_sum)
    b.y += ny * overlap * (inv_b / inv_sum)
    # Impulse along the normal.
    rvx = b.vx - a.vx
    rvy = b.vy - a.vy
    vel_along = rvx * nx + rvy * ny
    if vel_along > 0:
        return  # separating already
    e = physics.restitution
    j = -(1.0 + e) * vel_along / inv_sum
    j *= physics.momentum_transfer
    a.vx -= j * inv_a * nx
    a.vy -= j * inv_a * ny
    b.vx += j * inv_b * nx
    b.vy += j * inv_b * ny


def resolve_collisions(objects: list[ReactiveObject], physics: PhysicsConfig) -> None:
    n = len(objects)
    for i in range(n):
        for k in range(i + 1, n):
            _resolve_pair(objects[i], objects[k], physics)
