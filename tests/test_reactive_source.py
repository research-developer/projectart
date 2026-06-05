from projectart.detection.yolo_dots import Detection
from projectart.inputs.reactive import ReactiveSource
from projectart.reactive.config import ReactiveConfig


def _det(tid, cx, cy):
    return Detection(class_id=0, cx=cx, cy=cy, w=64, h=64, confidence=0.9,
                     class_name="person", track_id=tid)


def _cfg():
    return ReactiveConfig.from_dict({
        "tracking": {"confirm_after_hits": 1},
        "objects": {"box": {"radius": 0.06, "shape": "box", "kinematic": True, "color": "auto"}},
        "spawn": [],
        "rules": [{"match": {"class": "person"},
                   "action": {"spawn": "box", "behaviors": [{"follow": {"gain": 1.0}}]}}],
        "sim": {"tick_hz": 30},
    })


def test_step_emits_scene_frame_with_box_in_world_coords():
    src = ReactiveSource.for_testing(_cfg(), frame_w=640, frame_h=360)
    frame = src.step([_det(1, 320, 180)], ts=0.0)
    assert frame.type == "scene_frame"
    box = [o for o in frame.objects if o.kind == "box"][0]
    # 320/640, 180/360 -> centre of world
    assert abs(box.x - 0.5) < 1e-6 and abs(box.y - 0.5) < 1e-6
    assert box.track_id == 1


def test_step_keeps_same_object_id_across_frames():
    src = ReactiveSource.for_testing(_cfg(), frame_w=640, frame_h=360)
    f1 = src.step([_det(1, 100, 100)], ts=0.0)
    id1 = f1.objects[0].id
    f2 = src.step([_det(1, 500, 300)], ts=0.05)
    assert f2.objects[0].id == id1   # morph, not recreate


def test_velocity_computed_on_second_frame():
    src = ReactiveSource.for_testing(_cfg(), frame_w=640, frame_h=360)
    src.step([_det(1, 320, 180)], ts=0.0)        # first frame: vx=vy=0
    f2 = src.step([_det(1, 640, 360)], ts=1.0)   # moved (0.5,0.5) world in 1s
    box = f2.objects[0]
    assert abs(box.vx - 0.5) < 1e-4
    assert abs(box.vy - 0.5) < 1e-4
