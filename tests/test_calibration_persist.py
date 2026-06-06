from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from projectart.calibration.persist import (
    CALIB_VERSION,
    CalibrationDoc,
    CameraIntrinsics,
    StageSchema,
    StereoExtrinsics,
    UvBasisSchema,
    WallPlaneSchema,
    doc_with_stage,
    empty_for_canvas,
    load_calibration,
    save_calibration,
    stage_calibration_from_doc,
)
from projectart.geometry.stage import StageCalibration


def test_empty_doc_round_trip(tmp_path: Path):
    doc = empty_for_canvas(1920, 1080)
    target = tmp_path / "calib.json"
    save_calibration(doc, path=target)
    assert target.exists()

    loaded = load_calibration(target)
    assert loaded is not None
    assert loaded.version == CALIB_VERSION
    assert loaded.canvas.w == 1920
    assert loaded.canvas.h == 1080
    assert loaded.camera_a is None


def test_full_round_trip(tmp_path: Path):
    doc = CalibrationDoc(
        canvas={"w": 1920, "h": 1080},
        camera_a=CameraIntrinsics(
            url="rtsp://10.0.0.33/ch0_1.h264",
            K=[[800, 0, 960], [0, 800, 540], [0, 0, 1]],
            dist=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        camera_b=CameraIntrinsics(
            url="rtsp://10.0.0.34/ch0_1.h264",
            K=[[800, 0, 960], [0, 800, 540], [0, 0, 1]],
            dist=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        stereo=StereoExtrinsics(
            R_a_to_b=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            t_a_to_b=[0.3, 0.0, 0.0],
        ),
        wall_plane=WallPlaneSchema(normal=[0, 0, 1], centroid=[0, 0, 2]),
        uv_basis=UvBasisSchema(u=[1, 0, 0], v=[0, 1, 0]),
        homography_uv_to_canvas=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    target = tmp_path / "calib.json"
    save_calibration(doc, path=target)
    loaded = load_calibration(target)
    assert loaded is not None
    assert loaded.camera_a is not None
    assert loaded.camera_a.url == "rtsp://10.0.0.33/ch0_1.h264"
    assert loaded.stereo is not None
    assert loaded.stereo.t_a_to_b == [0.3, 0.0, 0.0]


def test_load_returns_none_when_missing(tmp_path: Path):
    target = tmp_path / "does-not-exist.json"
    assert load_calibration(target) is None


def test_load_swallows_garbage(tmp_path: Path):
    target = tmp_path / "garbage.json"
    target.write_text("{ this is not json")
    assert load_calibration(target) is None


def test_load_rejects_unknown_keys(tmp_path: Path):
    target = tmp_path / "extra.json"
    payload = {
        "version": CALIB_VERSION,
        "canvas": {"w": 1920, "h": 1080},
        "totally_made_up": "yes",
    }
    target.write_text(json.dumps(payload))
    # extra='forbid' on the model means this load returns None (caught + logged)
    assert load_calibration(target) is None


def test_camera_intrinsics_shape_validation():
    with pytest.raises(ValidationError):
        # K must be 3x3 — providing 2x2 should fail
        CameraIntrinsics(url="x", K=[[1, 0], [0, 1]], dist=[0, 0, 0, 0])


def test_atomic_write_does_not_leave_tmp(tmp_path: Path):
    doc = empty_for_canvas(1280, 720)
    target = tmp_path / "calib.json"
    save_calibration(doc, path=target)
    siblings = list(tmp_path.iterdir())
    assert all(p.suffix != ".tmp" for p in siblings), siblings


def test_stage_round_trips_through_doc():
    cal = StageCalibration.identity(640, 360, 1920, 1080)
    cal.height_offset = (0.01, -0.02)
    doc = doc_with_stage(CalibrationDoc(), cal)
    back = stage_calibration_from_doc(doc)
    assert back is not None
    assert np.allclose(back.cam_to_stage, cal.cam_to_stage)
    assert np.allclose(back.stage_to_projector, cal.stage_to_projector)
    assert back.height_offset == (0.01, -0.02)


def test_no_stage_section_returns_none():
    assert stage_calibration_from_doc(CalibrationDoc()) is None


def test_stage_schema_rejects_non_3x3():
    with pytest.raises(ValidationError):
        StageSchema(
            cam_to_stage=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            stage_to_projector=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
