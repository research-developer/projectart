"""1€ filter — adaptive low-pass for noisy interactive input.

Reference: Géry Casiez et al., "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems" (CHI 2012).

The filter raises its cutoff with hand velocity, so slow motion is smoothed
heavily (kills jitter) and fast motion stays responsive (preserves intent).
This is what we use on every position-bearing channel: dot tip, HUD anchor,
contact distance.

Defaults (`mincutoff=1.0, beta=0.05`) come from the drawmagic source PRD;
they're the well-tested "kid mode" preset.
"""
from __future__ import annotations

import math

import numpy as np


class _LowPass:
    __slots__ = ("y",)

    def __init__(self) -> None:
        self.y: float | np.ndarray | None = None

    def __call__(self, x, alpha: float):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Generic 1€ filter that works on scalars or numpy vectors.

    Use it like a function::

        f = OneEuroFilter(mincutoff=1.0, beta=0.05)
        smoothed = f(raw_xy, t_seconds)
    """

    __slots__ = ("mincutoff", "beta", "dcutoff", "_x_lp", "_dx_lp", "_t_prev", "_x_prev")

    def __init__(
        self,
        mincutoff: float = 1.0,
        beta: float = 0.05,
        dcutoff: float = 1.0,
    ):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_lp = _LowPass()
        self._dx_lp = _LowPass()
        self._t_prev: float | None = None
        self._x_prev = None

    def reset(self) -> None:
        self._x_lp = _LowPass()
        self._dx_lp = _LowPass()
        self._t_prev = None
        self._x_prev = None

    def __call__(self, x, t: float):
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = max(1e-3, t - self._t_prev)
        dx = (x - self._x_prev) / dt
        edx = self._dx_lp(dx, _alpha(self.dcutoff, dt))
        # magnitude of derivative — works for scalar (abs) and array (norm)
        if isinstance(edx, np.ndarray):
            mag = float(np.linalg.norm(edx))
        else:
            mag = abs(float(edx))
        cutoff = self.mincutoff + self.beta * mag
        ex = self._x_lp(x, _alpha(cutoff, dt))
        self._t_prev = t
        self._x_prev = x
        return ex
