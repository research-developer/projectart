"""Camera-pixel -> normalized world [0,1] coordinate mapping.

Pure numpy. Default is linear-by-frame-size (identity); a 3x3 homography can
be supplied later (projector calibration) without changing any caller — that
is the "mock now, real transform later" seam from the design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CamToWorld:
    matrix: np.ndarray | None = None  # 3x3 homography, or None for linear-by-size

    @classmethod
    def identity(cls) -> "CamToWorld":
        return cls(matrix=None)

    @classmethod
    def from_config(cls, spec) -> "CamToWorld":
        if spec is None or spec == "identity":
            return cls(matrix=None)
        m = np.asarray(spec, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"cam_to_world homography must be 3x3; got {m.shape}")
        return cls(matrix=m)

    def __call__(self, px: float, py: float, frame_w: int, frame_h: int) -> tuple[float, float]:
        if self.matrix is None:
            return (px / max(1, frame_w), py / max(1, frame_h))
        v = self.matrix @ np.array([px, py, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-12:
            return (0.0, 0.0)
        return (float(v[0] / v[2]), float(v[1] / v[2]))
