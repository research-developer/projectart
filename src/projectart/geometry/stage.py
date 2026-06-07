"""StageCalibration — camera px <-> normalized stage [0,1] <-> projector px.

Two 4-corner homographies plus a small residual height offset. cv2 is allowed
in geometry/; import it lazily so the module imports without OpenCV in odd envs.
The camera->stage homography is meant to be calibrated with the marker at
PLAYING HEIGHT, which makes it exact for that plane (see the design spec).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical stage corner order: top-left, top-right, bottom-right, bottom-left.
STAGE_CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _homography(src, dst) -> np.ndarray:
    import cv2

    s = np.asarray(src, dtype=np.float32)
    d = np.asarray(dst, dtype=np.float32)
    return cv2.getPerspectiveTransform(s, d).astype(np.float64)


def _apply(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = h @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-12:
        return (0.0, 0.0)
    return (float(v[0] / v[2]), float(v[1] / v[2]))


@dataclass(slots=True)
class StageCalibration:
    cam_to_stage: np.ndarray  # 3x3: camera px -> stage [0,1]
    stage_to_projector: np.ndarray  # 3x3: stage [0,1] -> projector px
    height_offset: tuple[float, float] = (0.0, 0.0)  # residual, stage units

    @classmethod
    def identity(cls, frame_w: int, frame_h: int, proj_w: int, proj_h: int) -> StageCalibration:
        c2s = np.array(
            [[1.0 / frame_w, 0, 0], [0, 1.0 / frame_h, 0], [0, 0, 1]], dtype=np.float64
        )
        s2p = np.array([[proj_w, 0, 0], [0, proj_h, 0], [0, 0, 1]], dtype=np.float64)
        return cls(cam_to_stage=c2s, stage_to_projector=s2p)

    @classmethod
    def from_corners(
        cls,
        cam_corners,
        stage_corners,
        proj_corners,
        height_offset: tuple[float, float] = (0.0, 0.0),
    ) -> StageCalibration:
        return cls(
            cam_to_stage=_homography(cam_corners, stage_corners),
            stage_to_projector=_homography(stage_corners, proj_corners),
            height_offset=tuple(height_offset),
        )

    def cam_px_to_stage(self, px: float, py: float) -> tuple[float, float]:
        sx, sy = _apply(self.cam_to_stage, px, py)
        return (sx + self.height_offset[0], sy + self.height_offset[1])

    def stage_to_proj_px(self, sx: float, sy: float) -> tuple[float, float]:
        return _apply(self.stage_to_projector, sx, sy)
