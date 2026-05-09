"""Smoke tests for the YOLO detector wrapper. We don't actually load the
network here — that requires downloading weights and pytorch. We just
exercise the dataclass and the path-resolution logic."""
from __future__ import annotations

from pathlib import Path

from projectart.detection.yolo_dots import Detection, DotDetector


def test_detection_dataclass():
    d = Detection(class_id=2, cx=100.0, cy=200.0, w=10.0, h=10.0, confidence=0.9)
    assert d.class_id == 2
    assert d.class_name == ""


def test_detector_unloaded_state():
    det = DotDetector(weights_path=Path("/does/not/exist.pt"))
    assert det._model is None
    assert det.weights_path == Path("/does/not/exist.pt")
