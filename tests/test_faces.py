"""Tests for the face gallery (matching + persistence). Pure numpy, no models."""
from __future__ import annotations

import numpy as np

from projectart.detection.faces import FaceGallery


def _vec(*head):
    v = np.zeros(128, dtype=np.float32)
    v[: len(head)] = head
    return v


def test_enroll_and_match_nearest():
    g = FaceGallery()
    g.enroll("Samaya", _vec(1, 0, 0))
    g.enroll("Preston Temple", _vec(0, 1, 0))
    q = _vec(1, 0, 0) + np.random.RandomState(0).normal(0, 0.01, 128).astype(np.float32)
    name, score = g.match(q, threshold=0.5)
    assert name == "Samaya"
    assert score > 0.9


def test_unknown_below_threshold():
    g = FaceGallery()
    g.enroll("Samaya", _vec(1, 0, 0))
    name, score = g.match(_vec(0, 0, 1), threshold=0.363)  # orthogonal -> no match
    assert name is None
    assert score < 0.363


def test_multiple_embeddings_per_person():
    g = FaceGallery()
    g.enroll("Samaya", _vec(1, 0, 0))
    g.enroll("Samaya", _vec(0, 1, 0))
    assert g.people["Samaya"].shape == (2, 128)
    name, _ = g.match(_vec(0, 1, 0), threshold=0.5)
    assert name == "Samaya"


def test_save_load_roundtrip(tmp_path):
    g = FaceGallery()
    g.enroll("Preston Temple", np.arange(128, dtype=np.float32))
    g.enroll("Samaya", np.ones(128, dtype=np.float32))
    path = tmp_path / "gallery.npz"
    g.save(path)
    g2 = FaceGallery.load(path)
    assert set(g2.names()) == {"Preston Temple", "Samaya"}
    assert np.allclose(g2.people["Samaya"], g.people["Samaya"])
    assert np.allclose(g2.people["Preston Temple"], g.people["Preston Temple"])
