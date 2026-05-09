"""Wire protocol between the Python backend and the browser renderer.

Messages are JSON. Every message has a `type` field. Future schema changes go
through `version` (current = 1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


PROTOCOL_VERSION = 1


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


def to_dict(event) -> dict:
    return asdict(event)
