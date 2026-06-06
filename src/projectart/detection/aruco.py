"""ArUco marker detection (cv2.aruco). Small dictionary (DICT_4X4_50) for speed
and robustness at distance/skew. ArUco ids are natively stable across frames, so
no association layer is needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Marker:
    id: int
    corners: np.ndarray  # (4,2) float, camera px
    cx: float
    cy: float


class ArucoDetector:
    def __init__(self, dictionary: str = "DICT_4X4_50"):
        self._dictionary = dictionary
        self._detector = None

    def _ensure(self) -> None:
        if self._detector is not None:
            return
        import cv2

        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self._dictionary))
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(d, params)

    def __call__(self, frame_bgr: np.ndarray) -> list[Marker]:
        self._ensure()
        corners, ids, _ = self._detector.detectMarkers(frame_bgr)
        if ids is None:
            return []
        out: list[Marker] = []
        for c, i in zip(corners, ids.flatten(), strict=True):
            pts = np.asarray(c, dtype=np.float64).reshape(4, 2)
            out.append(
                Marker(
                    id=int(i),
                    corners=pts,
                    cx=float(pts[:, 0].mean()),
                    cy=float(pts[:, 1].mean()),
                )
            )
        return out
