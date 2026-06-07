from __future__ import annotations

import numpy as np
import pytest

from projectart.geometry.wall_plane import (
    Plane,
    build_basis,
    fit_plane,
    is_in_contact,
    project_to_uv,
    ray_plane_intersect,
    signed_distance_to_plane,
)

# A reference plane: z = 1 (i.e. normal (0,0,1), centroid (0,0,1))
PLANE_Z1 = Plane(normal=np.array([0.0, 0.0, 1.0]), centroid=np.array([0.0, 0.0, 1.0]))


def test_fit_plane_xy_plane():
    """Points on z=1 should fit to a horizontal plane."""
    pts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    plane = fit_plane(pts)
    # Normal should be ±z
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-6
    assert abs(plane.normal[0]) < 1e-6
    assert abs(plane.normal[1]) < 1e-6
    # Centroid is at the mean
    np.testing.assert_allclose(plane.centroid, [0.5, 0.5, 1.0], atol=1e-9)


def test_fit_plane_with_noise():
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [
            rng.uniform(-1, 1, 50),
            rng.uniform(-1, 1, 50),
            np.full(50, 2.0) + rng.normal(0, 0.001, 50),  # ~z=2
        ]
    )
    plane = fit_plane(pts)
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-3


def test_fit_plane_requires_three_points():
    with pytest.raises(ValueError):
        fit_plane(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_fit_plane_requires_3d():
    with pytest.raises(ValueError):
        fit_plane(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]))


def test_signed_distance():
    above = np.array([0.5, 0.5, 1.5])  # above the plane in +normal direction
    below = np.array([0.5, 0.5, 0.5])
    assert signed_distance_to_plane(above, PLANE_Z1) > 0
    assert signed_distance_to_plane(below, PLANE_Z1) < 0
    on = np.array([0.5, 0.5, 1.0])
    assert abs(signed_distance_to_plane(on, PLANE_Z1)) < 1e-12


def test_is_in_contact():
    on = np.array([0.5, 0.5, 1.005])      # 5 mm above
    far = np.array([0.5, 0.5, 1.05])      # 5 cm above
    assert is_in_contact(on, PLANE_Z1, epsilon_m=0.015) is True
    assert is_in_contact(far, PLANE_Z1, epsilon_m=0.015) is False


def test_ray_plane_intersect_forward():
    # ray from origin pointing up, intersects z=1 at (0,0,1)
    o = np.array([0.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])
    hit = ray_plane_intersect(o, d, PLANE_Z1)
    np.testing.assert_allclose(hit, [0.0, 0.0, 1.0], atol=1e-9)


def test_ray_plane_intersect_backward_rejected():
    o = np.array([0.0, 0.0, 2.0])
    d = np.array([0.0, 0.0, 1.0])  # pointing away from plane at z=1
    assert ray_plane_intersect(o, d, PLANE_Z1, only_forward=True) is None
    # backward intersection allowed when explicitly enabled
    hit = ray_plane_intersect(o, d, PLANE_Z1, only_forward=False)
    np.testing.assert_allclose(hit, [0.0, 0.0, 1.0], atol=1e-9)


def test_ray_parallel_to_plane_returns_none():
    o = np.array([0.0, 0.0, 0.5])
    d = np.array([1.0, 0.0, 0.0])    # parallel to z=1 plane
    assert ray_plane_intersect(o, d, PLANE_Z1) is None


def test_basis_and_uv_round_trip():
    # Plane z=0; corners at (0,0,0) and (1,0,0)
    plane = Plane(normal=np.array([0.0, 0.0, 1.0]), centroid=np.array([0.5, 0.5, 0.0]))
    basis = build_basis(plane, p_topleft=[0.0, 0.0, 0.0], p_topright=[1.0, 0.0, 0.0])
    # u should be along +x
    np.testing.assert_allclose(basis.u, [1.0, 0.0, 0.0], atol=1e-9)
    # v should be along ±y
    assert abs(abs(basis.v[1]) - 1.0) < 1e-6

    # Round trip: a point at (0.5, 0.5, 0) is at the centroid → uv ~ (0,0)
    uv0 = project_to_uv(np.array([0.5, 0.5, 0.0]), basis)
    np.testing.assert_allclose(uv0, [0.0, 0.0], atol=1e-9)

    # An off-plane component should be dropped
    uv1 = project_to_uv(np.array([0.5, 0.5, 99.0]), basis)
    np.testing.assert_allclose(uv1, [0.0, 0.0], atol=1e-9)
