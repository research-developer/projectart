from __future__ import annotations

import numpy as np

from projectart.geometry.filter_one_euro import OneEuroFilter


def test_first_call_returns_input():
    f = OneEuroFilter()
    out = f(np.array([1.0, 2.0]), t=0.0)
    assert np.allclose(out, [1.0, 2.0])


def test_smooths_constant_signal():
    f = OneEuroFilter(mincutoff=1.0, beta=0.0)
    val = np.array([5.0, 5.0])
    out = f(val, t=0.0)
    for tick in range(1, 20):
        out = f(val, t=tick * 0.016)
    assert np.allclose(out, val, atol=0.01)


def test_high_velocity_passes_through():
    """With β > 0, fast motion increases the cutoff — output tracks the
    real signal more closely than for slow motion."""
    f = OneEuroFilter(mincutoff=0.5, beta=1.0)
    # rapid step from 0 to 100 — output should rise quickly
    f(np.array([0.0, 0.0]), t=0.0)
    out = None
    for tick in range(1, 6):
        out = f(np.array([100.0, 100.0]), t=tick * 0.016)
    assert out is not None
    # Should be well past the midpoint within ~80 ms
    assert out[0] > 60.0


def test_scalar_works():
    f = OneEuroFilter()
    f(0.0, 0.0)
    f(1.0, 0.016)
    out = f(2.0, 0.032)
    assert isinstance(out, float) or hasattr(out, "__float__")


def test_reset_clears_state():
    f = OneEuroFilter()
    f(np.array([10.0]), t=0.0)
    f(np.array([20.0]), t=0.016)
    f.reset()
    assert f._t_prev is None
    out = f(np.array([99.0]), t=0.0)
    assert np.allclose(out, [99.0])
