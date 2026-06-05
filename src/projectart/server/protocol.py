"""Wire protocol between the Python backend and the browser renderer.

Messages are JSON. Every message has a `type` field. Future schema changes go
through `version` (current = 1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

PROTOCOL_VERSION = 2


@dataclass
class Hello:
    type: Literal["hello"] = "hello"
    version: int = PROTOCOL_VERSION
    canvas_w: int = 1920
    canvas_h: int = 1080


@dataclass
class PointerEvent:
    """One point along a stroke. Backend emits these continuously while the
    pointer is active. Renderer connects two consecutive points with a stroke
    segment when both have `contact == True`."""
    x: float
    y: float
    contact: bool
    velocity: float = 0.0
    hand_id: int = 0       # 0 = right, 1 = left
    finger_id: int = 8     # MediaPipe-style: 4 = thumb tip, 8 = index tip
    confidence: float = 1.0
    ts_ms: int = 0
    type: Literal["pointer"] = "pointer"


@dataclass
class HudAnchorEvent:
    """Where the floating HUD should anchor itself (canvas px). Backend
    aggregates from dot detections; renderer tweens to soften jumps."""
    x: float
    y: float
    visible: bool = True
    ts_ms: int = 0
    type: Literal["hud_anchor"] = "hud_anchor"


@dataclass
class CommandEvent:
    """Discrete UI command from the backend (e.g. `undo`, `clear`, `mode_toggle`)."""
    command: str
    type: Literal["command"] = "command"


@dataclass
class EntityEvent:
    """One lifecycle frame for a tracked entity. Phase is `enter`, `update`,
    or `leave`. Bbox coordinates are in canvas pixels (mapped through the
    cam-to-canvas helper before broadcast). Renderer uses (track_id, phase)
    to drive overlay fade-in / fade-out."""
    track_id: int
    class_name: str
    phase: Literal["enter", "update", "leave"]
    bbox_x: float       # canvas px (top-left corner x)
    bbox_y: float       # canvas px (top-left corner y)
    bbox_w: float       # canvas px width
    bbox_h: float       # canvas px height
    confidence: float = 1.0
    ts_ms: int = 0
    type: Literal["entity"] = "entity"


@dataclass
class SceneObject:
    """One reactive object in a SceneFrame. World units: x,y,r in [0,1];
    vx,vy in world-units/sec. `shape` is a render hint (box|circle)."""

    id: int
    kind: str
    shape: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.05
    color: str = "#39f"
    state: str = ""
    alpha: float = 1.0
    angle: float = 0.0
    track_id: int | None = None


@dataclass
class SceneFrame:
    """A full snapshot of all reactive objects for one sim tick. The renderer
    keys visuals on object id and morphs between successive frames."""

    ts_ms: int
    objects: list = field(default_factory=list)
    type: Literal["scene_frame"] = "scene_frame"


def to_dict(event) -> dict:
    return asdict(event)
