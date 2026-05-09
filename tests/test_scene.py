"""Tests for the scene publisher's delta logic. We don't spin up
cameras / YOLO / WS — we drive `step()` directly with synthetic
detections and assert the EntityEvents we'd broadcast."""
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.inputs.scene import ScenePublisher
from projectart.server.ws import Server
from projectart.tracking.builtins import Cat, Person


def _det(class_id, name, cx=200.0, cy=200.0, w=40.0, h=40.0, conf=0.9):
    return Detection(class_id=class_id, cx=cx, cy=cy, w=w, h=h, confidence=conf, class_name=name)


def _make_publisher(canvas_size=(1920, 1080)) -> ScenePublisher:
    """Build a publisher without starting capture / detector. Tests drive
    `step()` directly so the camera + YOLO are never touched."""
    # Use tmp paths / placeholders for things we won't actually invoke.
    server = Server(
        ws_host="127.0.0.1",
        ws_port=0,
        http_port=0,
        static_dir=__import__("pathlib").Path("/tmp"),
        canvas_size=canvas_size,
    )
    pub = ScenePublisher(
        canvas_size=canvas_size,
        server=server,
        camera_url_a="rtsp://placeholder/cam-a",
        yolo_weights_path=None,
    )
    return pub


def test_first_frame_emits_enter():
    pub = _make_publisher()
    events = pub.step([_det(15, "cat")], ts=0.0)
    assert [e.phase for e in events] == ["enter"]
    e = events[0]
    assert e.class_name == "cat"
    assert e.bbox_w > 0 and e.bbox_h > 0


def test_followup_frame_emits_update_with_same_track_id():
    pub = _make_publisher()
    pub.step([_det(15, "cat", cx=200, cy=200)], ts=0.0)
    events = pub.step([_det(15, "cat", cx=205, cy=200)], ts=0.05)
    assert [e.phase for e in events] == ["update"]
    assert events[0].track_id == 1   # first allocated id


def test_disappearance_emits_leave():
    pub = _make_publisher()
    pub.step([_det(15, "cat")], ts=0.0)
    # Skip far enough to age past GONE_AFTER_S (default 2s)
    events = pub.step([], ts=5.0)
    phases = [e.phase for e in events]
    assert "leave" in phases


def test_two_classes_two_tracks():
    pub = _make_publisher()
    events = pub.step([_det(15, "cat", cx=100, cy=100), _det(0, "person", cx=900, cy=500)], ts=0.0)
    classes = sorted(e.class_name for e in events)
    assert classes == ["cat", "person"]
    assert len(set(e.track_id for e in events)) == 2


def test_canvas_mapping_scales_to_canvas_pixels():
    pub = _make_publisher(canvas_size=(640, 480))
    events = pub.step([_det(15, "cat", cx=960, cy=540, w=20, h=20)], ts=0.0)
    e = events[0]
    # Cam frame is assumed 1920x1080 in step(); cx=960 → canvas center x = 320
    # (top-left corner of bbox in canvas px = 320 - (20/2)*scale = ...)
    assert 300 < e.bbox_x + e.bbox_w / 2 < 340
    assert 220 < e.bbox_y + e.bbox_h / 2 < 260


def test_re_acquire_does_not_reissue_enter():
    """A briefly-missed entity that returns to PRESENT keeps its track_id
    and emits update, not a fresh enter."""
    pub = _make_publisher()
    pub.step([_det(15, "cat")], ts=0.0)
    pub.step([], ts=0.7)              # > LOST_AFTER_S → LEAVING but still tracked
    events = pub.step([_det(15, "cat")], ts=0.9)
    assert [e.phase for e in events] == ["update"]
