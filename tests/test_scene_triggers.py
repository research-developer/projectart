"""Integration: the scene source wires the registry -> TriggerEngine -> bus."""
from __future__ import annotations

from pathlib import Path

from projectart.detection.yolo_dots import Detection
from projectart.inputs.scene import ScenePublisher
from projectart.server.ws import Server


def _det(cid, name, cx=300.0, cy=300.0, w=80.0, h=80.0):
    return Detection(class_id=cid, cx=cx, cy=cy, w=w, h=h, confidence=0.9, class_name=name)


def _scene():
    server = Server(ws_host="127.0.0.1", ws_port=0, http_port=0,
                    static_dir=Path("."), canvas_size=(1920, 1080))
    # audio + face recognition off so the test is hermetic (no afplay, no model
    # download, no ~/.projectart gallery dependency)
    return ScenePublisher(canvas_size=(1920, 1080), server=server,
                          camera_url_a="x", enable_cat_audio=False,
                          enable_face_recognition=False)


def test_scene_emits_cat_appear_event_on_bus():
    sp = _scene()
    got = []
    sp.bus.on("event.appear", lambda *, event: got.append(event))
    # confirm_after_hits=2 -> needs two frames before the cat is confirmed
    for ts in (0.0, 0.1):
        sp.registry.consume([_det(15, "cat")], ts=ts)
        sp.triggers.update(sp.registry, ts, 640 * 360)
    assert any(e.kind == "appear" and e.class_name == "cat" for e in got)


def test_scene_emits_person_cat_intersect_event_on_bus():
    sp = _scene()
    got = []
    sp.bus.on("event.intersect", lambda *, event: got.append(event))
    for ts in (0.0, 0.1):
        sp.registry.consume(
            [_det(0, "person", cx=300, cy=300, w=300, h=300),
             _det(15, "cat", cx=300, cy=300, w=60, h=60)],
            ts=ts,
        )
        sp.triggers.update(sp.registry, ts, 640 * 360)
    assert any(e.kind == "intersect" for e in got)


def _appear_and_name(sp, name="Samaya", cx=300, cy=300):
    """Confirm a person entity and assign it a recognized face name."""
    face = ((cx - 10, cy - 10, 20, 20), name, 0.5)  # face centre inside the person box
    for ts in (0.0, 0.1):
        sp.registry.consume([_det(0, "person", cx=cx, cy=cy, w=200, h=200)], ts=ts)
        sp.triggers.update(sp.registry, ts, 640 * 360)
    from projectart.inputs.scene import _assign_face_names
    _assign_face_names(sp.registry, [face])


def test_scene_face_name_greets_after_dwell_not_immediately():
    """End-to-end: _assign_face_names tags the person, but the greet waits for the
    dwell (no premature greeting) — then RecognizeTrigger fires once."""
    sp = _scene()
    got = []
    sp.bus.on("event.recognize", lambda *, event: got.append(event))
    _appear_and_name(sp, "Samaya")
    sp.triggers.update(sp.registry, 0.2, 640 * 360)
    assert not got  # named, but within the greet dwell -> not greeted yet
    # still tracked, now past the dwell -> greet exactly once
    sp.registry.consume([_det(0, "person", cx=300, cy=300, w=200, h=200)], ts=2.0)
    sp.triggers.update(sp.registry, 2.0, 640 * 360)
    assert [(e.kind, e.name) for e in got] == [("recognize", "Samaya")]


def test_scene_says_farewell_after_recognized_person_gone():
    """A greeted person who disappears past the gone threshold gets a farewell."""
    sp = _scene()
    fare = []
    sp.bus.on("event.farewell", lambda *, event: fare.append(event))
    _appear_and_name(sp, "Samaya")
    sp.registry.consume([_det(0, "person", cx=300, cy=300, w=200, h=200)], ts=2.0)
    sp.triggers.update(sp.registry, 2.0, 640 * 360)  # past dwell -> eligible (greeted)
    assert not fare
    # person leaves; advance past gone_after_s (3.0) so the registry drops them
    sp.registry.consume([], ts=6.0)
    sp.triggers.update(sp.registry, 6.0, 640 * 360)
    assert any(e.kind == "farewell" and e.name == "Samaya" for e in fare)
