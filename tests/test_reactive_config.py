from __future__ import annotations

from projectart.reactive.config import ReactiveConfig, load_default


def test_default_config_loads_and_has_box_and_ball():
    cfg = load_default()
    assert "box" in cfg.objects and "ball" in cfg.objects
    assert cfg.objects["box"].kinematic is True
    assert cfg.objects["ball"].kinematic is False
    assert cfg.sim.tick_hz > 0
    assert len(cfg.rules) >= 1


def test_from_dict_round_trip_minimal():
    d = {
        "tracking": {"confirm_after_hits": 5},
        "objects": {"ball": {"radius": 0.1, "shape": "circle", "mass": 2.0, "kinematic": False}},
        "spawn": [{"kind": "ball", "x": 0.5, "y": 0.5}],
        "rules": [{"match": {"class": "person"}, "action": {"spawn": "ball", "behaviors": []}}],
        "physics": {"restitution": 0.5},
        "sim": {"tick_hz": 60},
    }
    cfg = ReactiveConfig.from_dict(d)
    assert cfg.tracking.confirm_after_hits == 5
    assert cfg.objects["ball"].radius == 0.1
    assert cfg.physics.restitution == 0.5
    assert cfg.sim.tick_hz == 60
    assert cfg.rules[0].action["spawn"] == "ball"
