"""Smoke tests for the YOLO detector wrapper. We don't actually load the
network here — that requires downloading weights and pytorch. We just
exercise the dataclass and the path-resolution logic."""
from __future__ import annotations

from pathlib import Path

from projectart.detection.yolo_dots import Detection, DotDetector, _parse_boxes


def test_detection_dataclass():
    d = Detection(class_id=2, cx=100.0, cy=200.0, w=10.0, h=10.0, confidence=0.9)
    assert d.class_id == 2
    assert d.class_name == ""


def test_detector_unloaded_state():
    det = DotDetector(weights_path=Path("/does/not/exist.pt"))
    assert det._model is None
    assert det.weights_path == Path("/does/not/exist.pt")


class _FakeTensor:
    def __init__(self, arr):
        import numpy as np
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeBoxes:
    def __init__(self, xywh, cls, conf, ids=None):
        self.xywh = _FakeTensor(xywh)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)
        self.id = _FakeTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.xywh.numpy())


def test_parse_boxes_with_track_ids():
    boxes = _FakeBoxes(
        xywh=[[100, 100, 20, 20], [200, 200, 30, 30]],
        cls=[0, 15], conf=[0.9, 0.8], ids=[7, 9],
    )
    dets = _parse_boxes(boxes, {0: "person", 15: "cat"}, with_ids=True)
    assert [d.track_id for d in dets] == [7, 9]
    assert [d.class_name for d in dets] == ["person", "cat"]
    assert dets[0].cx == 100


def test_parse_boxes_without_ids():
    boxes = _FakeBoxes(xywh=[[1, 2, 3, 4]], cls=[0], conf=[0.5], ids=None)
    dets = _parse_boxes(boxes, {0: "person"}, with_ids=True)
    assert dets[0].track_id is None
