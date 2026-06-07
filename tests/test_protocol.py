"""Wire-format unit tests. These run without a display, camera, or browser."""
from __future__ import annotations

import json

from projectart.server.protocol import (
    PROTOCOL_VERSION,
    CommandEvent,
    Hello,
    HudAnchorEvent,
    PointerEvent,
    SceneFrame,
    SceneObject,
    to_dict,
)


def test_hello_defaults():
    h = Hello()
    d = to_dict(h)
    assert d["type"] == "hello"
    assert d["version"] == PROTOCOL_VERSION
    assert d["canvas_w"] == 1920
    assert d["canvas_h"] == 1080


def test_pointer_event_roundtrip():
    p = PointerEvent(x=10.0, y=20.0, contact=True, velocity=5.0, ts_ms=12345)
    d = to_dict(p)
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["type"] == "pointer"
    assert parsed["x"] == 10.0
    assert parsed["contact"] is True
    assert parsed["finger_id"] == 8  # default = index tip


def test_hud_anchor_event():
    h = HudAnchorEvent(x=100.0, y=200.0, visible=False, ts_ms=7)
    d = to_dict(h)
    assert d["type"] == "hud_anchor"
    assert d["visible"] is False


def test_command_event():
    c = CommandEvent(command="undo")
    d = to_dict(c)
    assert d == {"type": "command", "command": "undo"}


def test_protocol_version_is_2():
    assert PROTOCOL_VERSION == 2


def test_scene_frame_round_trips():
    f = SceneFrame(ts_ms=123, objects=[
        SceneObject(id=1, kind="box", shape="box", x=0.5, y=0.5, vx=0.1, vy=0.0,
                    r=0.06, color="#39f", state="", alpha=1.0, angle=0.0, track_id=7),
    ])
    d = to_dict(f)
    assert d["type"] == "scene_frame"
    assert d["objects"][0]["kind"] == "box"
    assert d["objects"][0]["track_id"] == 7
    assert d["objects"][0]["x"] == 0.5
