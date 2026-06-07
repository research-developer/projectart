"""ReactiveConfig — the single JSON-serialisable config tree.

This is the game-engine-portable contract: tracking knobs, world/coordinate
mapping, physics constants, object templates, spawn list, and match->action
rules. No behaviour is hard-coded; everything tunable lives here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..tracking.config import TrackingConfig


@dataclass(slots=True)
class WorldConfig:
    aspect: float = 16 / 9
    cam_to_world: str | list = "identity"  # "identity" or a 3x3 list (homography)


@dataclass(slots=True)
class PhysicsConfig:
    friction: float = 0.8
    restitution: float = 0.9
    # impulse scale: 1.0=normal, >1 super-elastic (artistic); must be >=0
    momentum_transfer: float = 1.0
    gravity: tuple[float, float] = (0.0, 0.0)
    default_mass: float = 1.0


@dataclass(slots=True)
class ObjectTemplate:
    kind: str
    radius: float = 0.05
    shape: str = "circle"  # box | circle
    mass: float = 1.0
    color: str = "#39f"
    kinematic: bool = False


@dataclass(slots=True)
class Rule:
    match: dict
    action: dict


@dataclass(slots=True)
class SimConfig:
    tick_hz: int = 30


@dataclass(slots=True)
class ReactiveConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    objects: dict[str, ObjectTemplate] = field(default_factory=dict)
    spawn: list[dict] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    sim: SimConfig = field(default_factory=SimConfig)
    despawn_ms: float = 600.0

    @classmethod
    def from_dict(cls, d: dict) -> ReactiveConfig:
        w = d.get("world", {})
        p = d.get("physics", {})
        s = d.get("sim", {})
        objects = {
            name: ObjectTemplate(
                kind=name,
                radius=float(o.get("radius", 0.05)),
                shape=o.get("shape", "circle"),
                mass=float(o.get("mass", 1.0)),
                color=o.get("color", "#39f"),
                kinematic=bool(o.get("kinematic", False)),
            )
            for name, o in d.get("objects", {}).items()
        }
        return cls(
            tracking=TrackingConfig.from_dict(d.get("tracking")),
            world=WorldConfig(
                aspect=float(w.get("aspect", 16 / 9)),
                cam_to_world=w.get("cam_to_world", "identity"),
            ),
            physics=PhysicsConfig(
                friction=float(p.get("friction", 0.8)),
                restitution=float(p.get("restitution", 0.9)),
                momentum_transfer=max(0.0, float(p.get("momentum_transfer", 1.0))),
                gravity=tuple(float(v) for v in p.get("gravity", (0.0, 0.0))),
                default_mass=float(p.get("default_mass", 1.0)),
            ),
            objects=objects,
            spawn=list(d.get("spawn", [])),
            rules=[
                Rule(match=r.get("match", {}), action=r.get("action", {}))
                for r in d.get("rules", [])
            ],
            sim=SimConfig(tick_hz=int(s.get("tick_hz", 30))),
            despawn_ms=float(d.get("despawn_ms", 600.0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ReactiveConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_default() -> ReactiveConfig:
    return ReactiveConfig.from_json(Path(__file__).with_name("default_config.json"))
