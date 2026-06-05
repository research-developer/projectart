"""Simulator — the backend reference simulation.

tick(entities, dt): bind/spawn objects from rules, drive kinematic (track-bound)
objects from entity motion + behaviors, despawn objects whose track left,
integrate dynamic objects, resolve collisions, return the object snapshot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .behaviors import EntityView, apply_behavior
from .config import ReactiveConfig
from .objects import ReactiveObject
from .physics import integrate, resolve_collisions
from .rules import match_rule, parse_behaviors

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackedView:
    track_id: int
    view: EntityView


def _hash_color(name: str) -> str:
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return f"hsl({h % 360}, 75%, 65%)"


class Simulator:
    def __init__(self, config: ReactiveConfig):
        self.config = config
        self.objects: list[ReactiveObject] = []
        self._next_id = 1
        self._bound: dict[int, int] = {}  # track_id -> object id
        self._despawn_age: dict[int, float] = {}  # object id -> seconds despawning
        self._spawn_static()

    def _spawn_static(self) -> None:
        for spec in self.config.spawn:
            t = self.config.objects.get(spec["kind"])
            if t is None:
                log.warning("spawn references unknown kind %r", spec.get("kind"))
                continue
            self.objects.append(
                ReactiveObject.from_template(
                    self._new_id(), t, x=float(spec.get("x", 0.5)), y=float(spec.get("y", 0.5))
                )
            )

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def tick(self, entities: list[TrackedView], dt: float) -> list[ReactiveObject]:
        seen = {tv.track_id for tv in entities}

        for tv in entities:
            obj = self._object_for_track(tv)
            if obj is None:
                continue
            if obj.kinematic:
                # velocity = entity world velocity (drives momentum transfer)
                obj.vx, obj.vy = tv.view.vx, tv.view.vy
            for beh in obj.behaviors:
                apply_behavior(obj, beh, tv.view)
            if obj.color == "auto":
                obj.color = _hash_color(tv.view.class_name)
            obj.age_s += dt

        # despawn bound objects whose track is gone
        for track_id, obj_id in list(self._bound.items()):
            if track_id in seen:
                continue
            obj = self._by_id(obj_id)
            if obj is None:
                self._bound.pop(track_id, None)
                continue
            obj.despawning = True
            age = self._despawn_age.get(obj_id, 0.0) + dt
            self._despawn_age[obj_id] = age
            frac = age / max(1e-3, self.config.despawn_ms / 1000.0)
            obj.alpha = max(0.0, 1.0 - frac)
            if obj.alpha <= 0.01:
                self.objects.remove(obj)
                self._bound.pop(track_id, None)
                self._despawn_age.pop(obj_id, None)

        for obj in self.objects:
            integrate(obj, dt, self.config.physics)
        resolve_collisions(self.objects, self.config.physics)
        return list(self.objects)

    def _object_for_track(self, tv: TrackedView) -> ReactiveObject | None:
        obj_id = self._bound.get(tv.track_id)
        if obj_id is not None:
            obj = self._by_id(obj_id)
            if obj is not None and obj.despawning:
                obj.despawning = False
                obj.alpha = 1.0
                self._despawn_age.pop(obj_id, None)
            return obj
        rule = match_rule(self.config.rules, tv.view.class_name)
        if rule is None:
            return None
        kind = rule.action.get("spawn")
        t = self.config.objects.get(kind)
        if t is None:
            log.warning("rule spawns unknown kind %r", kind)
            return None
        behaviors = parse_behaviors(rule.action.get("behaviors", []))
        obj = ReactiveObject.from_template(
            self._new_id(),
            t,
            x=tv.view.x,
            y=tv.view.y,
            bound_track_id=tv.track_id,
            behaviors=behaviors,
        )
        self.objects.append(obj)
        self._bound[tv.track_id] = obj.id
        return obj

    def _by_id(self, obj_id: int) -> ReactiveObject | None:
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None
