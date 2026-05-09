"""Stereo tests. The triangulation path needs cv2; if cv2 isn't installed
(e.g. a stripped CI), the cv2-dependent tests are skipped.

The correspondence helper is cv2-free and always tested."""
from __future__ import annotations

import numpy as np
import pytest

from projectart.detection.stereo import (
    StereoRig,
    correspond_by_class,
    triangulate_all,
    triangulate_pair,
)
from projectart.detection.yolo_dots import Detection


def _det(cid, cx, cy, conf=0.9):
    return Detection(class_id=cid, cx=cx, cy=cy, w=10, h=10, confidence=conf)


def test_correspond_by_class_simple():
    a = [_det(0, 100, 100), _det(1, 200, 200), _det(2, 300, 300)]
    b = [_det(1, 250, 200), _det(2, 350, 300), _det(3, 99, 99)]
    pairs = correspond_by_class(a, b)
    cids = [pa.class_id for pa, _ in pairs]
    assert cids == [1, 2]


def test_correspond_picks_highest_confidence_within_view():
    a = [_det(0, 0, 0, 0.4), _det(0, 0, 0, 0.9), _det(0, 0, 0, 0.5)]
    b = [_det(0, 0, 0, 0.6)]
    pairs = correspond_by_class(a, b)
    assert len(pairs) == 1
    assert pairs[0][0].confidence == pytest.approx(0.9)


def test_correspond_empty():
    assert correspond_by_class([], []) == []
    assert correspond_by_class([_det(0, 0, 0)], []) == []
    assert correspond_by_class([], [_det(0, 0, 0)]) == []


# ---- cv2-dependent triangulation ----

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

needs_cv2 = pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")


def _rectified_rig(baseline_m: float = 0.30) -> StereoRig:
    """A clean horizontal stereo rig: identical intrinsics, cam B shifted +x by baseline."""
    K = np.array(
        [
            [800.0, 0.0, 960.0],
            [0.0, 800.0, 540.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = np.eye(3)
    # Translation of A's origin in B's frame: cam B sees A at (-baseline, 0, 0)
    t = np.array([-baseline_m, 0.0, 0.0])
    return StereoRig(K_a=K, K_b=K, R_a_to_b=R, t_a_to_b=t)


@needs_cv2
def test_triangulate_known_point_round_trip():
    """Synthesize correspondences from a known 3D point and verify the
    triangulator recovers it within sub-mm tolerance."""
    rig = _rectified_rig(baseline_m=0.30)
    target = np.array([0.10, -0.05, 2.00])  # in cam-A space, 2 m down-range

    # Project to cam A: u = fx*X/Z + cx; v = fy*Y/Z + cy
    K = rig.K_a
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ua = fx * target[0] / target[2] + cx
    va = fy * target[1] / target[2] + cy

    # Cam-B sees the point at A-frame coords + (-t_a_to_b) since R=I
    # Equivalently: in B's frame, the point is at R*X + t_a_to_b
    Xb = rig.R_a_to_b @ target + rig.t_a_to_b
    ub = fx * Xb[0] / Xb[2] + cx
    vb = fy * Xb[1] / Xb[2] + cy

    recovered = triangulate_pair(rig, (ua, va), (ub, vb))
    np.testing.assert_allclose(recovered, target, atol=1e-3)


@needs_cv2
def test_triangulate_all_filters_degenerate():
    rig = _rectified_rig()
    a = [_det(0, 100, 100), _det(1, 200, 200)]
    b = [_det(0, 110, 100), _det(1, 220, 200)]
    pairs = correspond_by_class(a, b)
    pts = triangulate_all(rig, pairs)
    assert len(pts) == 2
    for p in pts:
        assert np.all(np.isfinite(p.point_3d))
