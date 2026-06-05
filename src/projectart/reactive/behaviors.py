"""Parameterized reactive behaviors. Pure functions of (object, params, entity).

A behavior is a tuple ``(name, params)``. The simulator applies the behavior
list of each object every tick. Adding a behavior = adding one branch here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .objects import ReactiveObject


@dataclass(slots=True)
class EntityView:
    """Read-only snapshot of a tracked entity in world space, passed to behaviors."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    bbox_area: float = 0.0  # fraction of frame area [0,1]
    confidence: float = 1.0
    dwell_s: float = 0.0
    class_name: str = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def apply_behavior(
    obj: ReactiveObject, behavior: tuple[str, dict], entity: EntityView | None
) -> None:
    name, params = behavior
    if name == "follow":
        if entity is None:
            return
        gain = float(params.get("gain", 0.5))
        obj.x += (entity.x - obj.x) * gain
        obj.y += (entity.y - obj.y) * gain
    elif name == "scale":
        if entity is None:
            return
        lo = float(params.get("min", 0.03))
        hi = float(params.get("max", 0.12))
        src = params.get("source", "bbox")
        metric = entity.bbox_area if src == "bbox" else entity.confidence
        obj.radius = _clamp(lo + (hi - lo) * _clamp(metric, 0.0, 1.0), lo, hi)
    elif name == "colorize":
        if entity is None:
            return
        mapping = params.get("mapping", "speed_to_hue")
        if mapping == "speed_to_hue":
            speed = math.hypot(entity.vx, entity.vy)
            hue = int(_clamp(speed, 0.0, 1.0) * 270)  # 0 (slow, red) .. 270 (fast, violet)
            obj.color = f"hsl({hue}, 80%, 60%)"
            obj.state = "fast" if speed > 0.3 else "slow"
    # unknown behaviors are silently ignored (forward-compatible config)
