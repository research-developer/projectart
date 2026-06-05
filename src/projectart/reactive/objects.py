"""ReactiveObject — one simulated object (world coords, [0,1])."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import ObjectTemplate


@dataclass(slots=True)
class ReactiveObject:
    id: int
    kind: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.05
    shape: str = "circle"
    mass: float = 1.0
    color: str = "#39f"
    state: str = ""
    alpha: float = 1.0
    angle: float = 0.0
    kinematic: bool = False
    bound_track_id: int | None = None
    age_s: float = 0.0
    behaviors: list = field(default_factory=list)  # list of (name, params)
    despawning: bool = False

    @property
    def inv_mass(self) -> float:
        if self.kinematic or self.mass <= 0:
            return 0.0
        return 1.0 / self.mass

    @classmethod
    def from_template(
        cls,
        obj_id: int,
        t: ObjectTemplate,
        x: float,
        y: float,
        bound_track_id: int | None = None,
        behaviors: list | None = None,
    ) -> ReactiveObject:
        return cls(
            id=obj_id,
            kind=t.kind,
            x=x,
            y=y,
            radius=t.radius,
            shape=t.shape,
            mass=t.mass,
            color=t.color,
            kinematic=t.kinematic,
            bound_track_id=bound_track_id,
            behaviors=behaviors or [],
        )
