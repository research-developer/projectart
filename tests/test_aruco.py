import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _marker_image(marker_id: int, size: int = 200, border: int = 60):
    import cv2 as _cv2

    d = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_4X4_50)
    img = _cv2.aruco.generateImageMarker(d, marker_id, size)
    canvas = np.full((size + 2 * border, size + 2 * border), 255, dtype=np.uint8)
    canvas[border : border + size, border : border + size] = img
    return _cv2.cvtColor(canvas, _cv2.COLOR_GRAY2BGR)


def test_detects_marker_id_and_center():
    from projectart.detection.aruco import ArucoDetector

    img = _marker_image(7)
    markers = ArucoDetector()(img)
    assert len(markers) == 1
    assert markers[0].id == 7
    h, w = img.shape[:2]
    assert markers[0].cx == pytest.approx(w / 2, abs=3)
    assert markers[0].cy == pytest.approx(h / 2, abs=3)


def test_no_markers_returns_empty():
    from projectart.detection.aruco import ArucoDetector

    blank = np.full((240, 320, 3), 127, dtype=np.uint8)
    assert ArucoDetector()(blank) == []
