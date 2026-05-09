"""Unit tests for the gloves source's calibration-loading helpers.

The full GlovesSource needs cameras + YOLO + threads — out of scope here.
But the calib → StereoRig / wall plane mapping is pure data and worth a
test, since "calibration loaded but stereo not active" was a real bug
class during the M2 → M3 transition."""
from __future__ import annotations

import numpy as np

from projectart.calibration.persist import (
    CalibrationDoc,
    CameraIntrinsics,
    StereoExtrinsics,
    UvBasisSchema,
    WallPlaneSchema,
)
from projectart.inputs.gloves import _stereo_rig_from_calib, _wall_plane_from_calib


def _full_calib() -> CalibrationDoc:
    return CalibrationDoc(
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
            t_a_to_b=[-0.30, 0.0, 0.0],
        ),
        wall_plane=WallPlaneSchema(normal=[0, 0, 1], centroid=[0, 0, 2]),
        uv_basis=UvBasisSchema(u=[1, 0, 0], v=[0, 1, 0]),
    )


def test_full_calib_yields_rig_and_plane():
    calib = _full_calib()
    rig = _stereo_rig_from_calib(calib)
    assert rig is not None
    assert rig.K_a.shape == (3, 3)
    assert rig.t_a_to_b.shape == (3,)
    np.testing.assert_allclose(rig.t_a_to_b, [-0.30, 0.0, 0.0])

    out = _wall_plane_from_calib(calib)
    assert out is not None
    plane, basis = out
    np.testing.assert_allclose(plane.normal, [0, 0, 1])
    np.testing.assert_allclose(plane.centroid, [0, 0, 2])
    assert basis is not None
    np.testing.assert_allclose(basis.u, [1, 0, 0])


def test_partial_calib_returns_none_for_rig_when_stereo_missing():
    calib = _full_calib()
    calib.stereo = None
    assert _stereo_rig_from_calib(calib) is None
    # Wall plane still works
    assert _wall_plane_from_calib(calib) is not None


def test_partial_calib_returns_none_for_plane_when_wall_missing():
    calib = _full_calib()
    calib.wall_plane = None
    assert _wall_plane_from_calib(calib) is None
    # Stereo rig still works
    assert _stereo_rig_from_calib(calib) is not None


def test_calib_with_no_uv_basis_still_returns_plane():
    calib = _full_calib()
    calib.uv_basis = None
    out = _wall_plane_from_calib(calib)
    assert out is not None
    plane, basis = out
    assert basis is None    # plane present, basis absent
