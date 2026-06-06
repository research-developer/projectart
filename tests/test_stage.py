from __future__ import annotations

import pytest

from projectart.geometry.stage import STAGE_CORNERS, StageCalibration


def test_identity_maps_frame_to_unit_and_unit_to_projector():
    cal = StageCalibration.identity(frame_w=640, frame_h=360, proj_w=1920, proj_h=1080)
    assert cal.cam_px_to_stage(320, 180) == pytest.approx((0.5, 0.5))
    assert cal.stage_to_proj_px(0.5, 0.5) == pytest.approx((960, 540))


def test_from_corners_maps_corners_exactly():
    # camera sees the stage corners at these pixels (TL,TR,BR,BL):
    cam = [(100, 80), (540, 90), (560, 300), (90, 290)]
    proj = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
    cal = StageCalibration.from_corners(cam, STAGE_CORNERS, proj)
    for (px, py), (sx, sy) in zip(cam, STAGE_CORNERS, strict=False):
        gx, gy = cal.cam_px_to_stage(px, py)
        assert (gx, gy) == pytest.approx((sx, sy), abs=1e-6)


def test_edge_midpoint_maps_where_predicted_through_composition():
    # A pure scaling homography: cam px = stage*1000. Midpoint of top edge
    # (stage 0.5,0) must come back to stage (0.5,0) — guards affine mistakes.
    cam = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    proj = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
    cal = StageCalibration.from_corners(cam, STAGE_CORNERS, proj)
    assert cal.cam_px_to_stage(500, 0) == pytest.approx((0.5, 0.0), abs=1e-6)
    assert cal.cam_px_to_stage(1000, 500) == pytest.approx((1.0, 0.5), abs=1e-6)


def test_height_offset_is_added_residual():
    cal = StageCalibration.identity(640, 360, 1920, 1080)
    cal.height_offset = (0.02, -0.01)
    gx, gy = cal.cam_px_to_stage(320, 180)
    assert (gx, gy) == pytest.approx((0.52, 0.49))
