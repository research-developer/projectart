"""Wall-plane geometry: fit a plane from 3D points, build an in-plane 2D
basis, intersect rays with the plane, measure signed distance.

Pure numpy, no cv2 — fully testable without OpenCV. Used by M3 to gate
"contact" (signed distance to wall < epsilon) and by M5 to set up the
homography from in-plane (u,v) coordinates to canvas pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Plane:
    """A 2D plane in 3D space, expressed as a unit normal and a centroid."""

    normal: np.ndarray   # (3,) float, unit length
    centroid: np.ndarray  # (3,) float


@dataclass(slots=True)
class PlaneBasis:
    """An orthonormal in-plane coordinate frame `(u, v)` together with the
    plane it spans. `u` and `v` are unit vectors in the plane; together with
    `plane.normal` they form a right-handed frame."""

    plane: Plane
    u: np.ndarray   # (3,) unit
    v: np.ndarray   # (3,) unit


def fit_plane(points: np.ndarray) -> Plane:
    """SVD plane fit. Requires `points.shape == (N, 3)` with `N >= 3`.

    The returned normal is unit-length but its sign is not pinned to any
    particular orientation. Callers that need a consistent sign should
    flip it themselves (e.g. so it points toward the cameras)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3); got {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError(f"need >=3 points to fit a plane; got {pts.shape[0]}")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    n = float(np.linalg.norm(normal))
    if n < 1e-12:
        raise ValueError("degenerate plane fit (collinear points?)")
    normal = normal / n
    return Plane(normal=normal.astype(np.float64), centroid=centroid.astype(np.float64))


def build_basis(plane: Plane, p_topleft: np.ndarray, p_topright: np.ndarray) -> PlaneBasis:
    """Build an in-plane (u,v) basis where `u` runs from top-left to top-right
    of the wall, projected into the plane. `v = normal × u` so the resulting
    frame is right-handed."""
    p_tl = np.asarray(p_topleft, dtype=np.float64).reshape(3)
    p_tr = np.asarray(p_topright, dtype=np.float64).reshape(3)
    u_raw = p_tr - p_tl
    # Remove any component along the normal so u lies in the plane
    u_in_plane = u_raw - u_raw.dot(plane.normal) * plane.normal
    u_norm = float(np.linalg.norm(u_in_plane))
    if u_norm < 1e-9:
        raise ValueError("topleft and topright collinear with the plane normal")
    u = u_in_plane / u_norm
    v = np.cross(plane.normal, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-9:
        raise ValueError("could not build orthogonal v (degenerate frame)")
    v = v / v_norm
    return PlaneBasis(plane=plane, u=u.astype(np.float64), v=v.astype(np.float64))


def project_to_uv(point_3d: np.ndarray, basis: PlaneBasis) -> np.ndarray:
    """Project a 3D point onto the plane and return its (u,v) coordinates
    in the basis frame. Projects orthogonally — the normal-component is
    dropped silently. Returns shape (2,) float64."""
    p = np.asarray(point_3d, dtype=np.float64).reshape(3)
    d = p - basis.plane.centroid
    return np.array([d.dot(basis.u), d.dot(basis.v)], dtype=np.float64)


def signed_distance_to_plane(point_3d: np.ndarray, plane: Plane) -> float:
    """Signed distance from a point to the plane, positive on the
    `+normal` side. Use the absolute value for "is in contact" checks."""
    p = np.asarray(point_3d, dtype=np.float64).reshape(3)
    return float((p - plane.centroid).dot(plane.normal))


def ray_plane_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: Plane,
    only_forward: bool = True,
) -> np.ndarray | None:
    """Intersect a ray (origin + t·direction, t>=0) with the plane.

    Returns the 3D intersection point, or None if the ray is parallel
    to the plane or (when `only_forward`) intersects behind the origin.
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    denom = float(d.dot(plane.normal))
    if abs(denom) < 1e-9:
        return None
    t = float((plane.centroid - o).dot(plane.normal) / denom)
    if only_forward and t < 0:
        return None
    return o + t * d


def is_in_contact(
    point_3d: np.ndarray,
    plane: Plane,
    epsilon_m: float = 0.015,
) -> bool:
    """True iff the absolute signed distance to the plane is below
    `epsilon_m`. Default 1.5 cm matches the spec's wall-contact threshold."""
    return abs(signed_distance_to_plane(point_3d, plane)) < epsilon_m
