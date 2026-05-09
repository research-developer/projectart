"""Stereo correspondence + triangulation.

Two YOLO detection lists (one per camera) → list of 3D dot positions in
cam-A coordinate space. Correspondence is by `class_id` — since dot gloves
use one YOLO class per finger, matching is trivial. If two dots in the
same view share a class_id (shouldn't happen with one glove), we keep the
highest-confidence one and warn.

cv2 is imported lazily so this module is importable in environments without
OpenCV (the dataclasses and matchers are cv2-free).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .yolo_dots import Detection

log = logging.getLogger(__name__)


@dataclass(slots=True)
class StereoRig:
    """Calibrated stereo pair. Convention: cam-A is the reference frame."""

    K_a: np.ndarray              # (3,3) intrinsics for cam A
    K_b: np.ndarray              # (3,3) intrinsics for cam B
    R_a_to_b: np.ndarray         # (3,3) rotation taking points in A's frame to B's frame
    t_a_to_b: np.ndarray         # (3,) translation in B's frame, "where A's origin is in B"
    dist_a: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    dist_b: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    def projection_a(self) -> np.ndarray:
        """3x4 projection matrix for cam A (identity extrinsics)."""
        return self.K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])

    def projection_b(self) -> np.ndarray:
        """3x4 projection matrix for cam B."""
        Rt = np.hstack([self.R_a_to_b, self.t_a_to_b.reshape(3, 1)])
        return self.K_b @ Rt


@dataclass(slots=True)
class TriangulatedDot:
    class_id: int
    point_3d: np.ndarray         # (3,) in cam-A frame
    confidence_a: float
    confidence_b: float
    px_a: tuple[float, float]
    px_b: tuple[float, float]
    class_name: str = ""

    @property
    def confidence(self) -> float:
        return min(self.confidence_a, self.confidence_b)


def correspond_by_class(
    dets_a: Iterable[Detection],
    dets_b: Iterable[Detection],
) -> list[tuple[Detection, Detection]]:
    """Match detections one-to-one by class_id, picking the highest-confidence
    candidate per class within each view. Returns the (a, b) pairs whose
    class_id appears in both views."""
    best_a: dict[int, Detection] = {}
    for d in dets_a:
        prev = best_a.get(d.class_id)
        if prev is None or d.confidence > prev.confidence:
            best_a[d.class_id] = d
    best_b: dict[int, Detection] = {}
    for d in dets_b:
        prev = best_b.get(d.class_id)
        if prev is None or d.confidence > prev.confidence:
            best_b[d.class_id] = d
    common = sorted(set(best_a) & set(best_b))
    return [(best_a[k], best_b[k]) for k in common]


def triangulate_pair(
    rig: StereoRig,
    px_a: tuple[float, float],
    px_b: tuple[float, float],
) -> np.ndarray:
    """Triangulate a single correspondence into a 3D point in cam-A space.

    Uses linear DLT via `cv2.triangulatePoints` followed by homogeneous
    normalization. For sub-pixel accuracy this is fine; if we ever need
    bundle-adjusted poses, swap to the iterative solver.
    """
    import cv2

    pa = rig.projection_a()
    pb = rig.projection_b()
    pts_a = np.array([[px_a[0]], [px_a[1]]], dtype=np.float64)
    pts_b = np.array([[px_b[0]], [px_b[1]]], dtype=np.float64)
    h = cv2.triangulatePoints(pa, pb, pts_a, pts_b)  # (4,1)
    w = float(h[3, 0])
    if abs(w) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return (h[:3, 0] / w).astype(np.float64)


def triangulate_all(
    rig: StereoRig,
    pairs: list[tuple[Detection, Detection]],
) -> list[TriangulatedDot]:
    """Triangulate every matched pair into a TriangulatedDot."""
    out: list[TriangulatedDot] = []
    for a, b in pairs:
        p3 = triangulate_pair(rig, (a.cx, a.cy), (b.cx, b.cy))
        if not np.all(np.isfinite(p3)):
            log.debug("dropping degenerate triangulation for class %d", a.class_id)
            continue
        out.append(
            TriangulatedDot(
                class_id=a.class_id,
                point_3d=p3,
                confidence_a=a.confidence,
                confidence_b=b.confidence,
                px_a=(a.cx, a.cy),
                px_b=(b.cx, b.cy),
                class_name=a.class_name or b.class_name,
            )
        )
    return out
