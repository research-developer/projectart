# tests/test_mapping.py
from __future__ import annotations

import pytest

from projectart.geometry.mapping import CamToWorld


def test_identity_divides_by_frame_size():
    m = CamToWorld.identity()
    assert m(320, 180, 640, 360) == pytest.approx((0.5, 0.5))
    assert m(0, 0, 640, 360) == pytest.approx((0.0, 0.0))
    assert m(640, 360, 640, 360) == pytest.approx((1.0, 1.0))


def test_from_config_identity_string():
    assert CamToWorld.from_config("identity").matrix is None
    assert CamToWorld.from_config(None).matrix is None


def test_homography_maps_pixels_to_world():
    # A homography that scales pixels by 1/1000 (so 500px -> 0.5 world).
    H = [[1 / 1000, 0, 0], [0, 1 / 1000, 0], [0, 0, 1]]
    m = CamToWorld.from_config(H)
    # frame size is ignored when a matrix is present
    assert m(500, 250, 640, 360) == pytest.approx((0.5, 0.25))


def test_bad_homography_shape_raises():
    with pytest.raises(ValueError):
        CamToWorld.from_config([[1, 0], [0, 1]])
