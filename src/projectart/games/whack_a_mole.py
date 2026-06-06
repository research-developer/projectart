"""Whack-a-Mole game logic (stage coordinates, [0,1]).

Pure and headless. Moles spawn on a grid on a fixed cadence and expire after a
lifetime. A whack is detected by the SWEPT SEGMENT of a marker's last->current
stage position passing within `hit_radius` of an active mole (so a fast swing
that jumps across the mole between frames still registers). Each mole scores at
most once (it is removed on hit).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def seg_point_dist(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


@dataclass(slots=True)
class WhackConfig:
    rows: int = 3
    cols: int = 3
    spawn_interval_s: float = 1.2
    mole_lifetime_s: float = 1.5
    points: int = 1
    round_seconds: float = 60.0
    hit_radius: float = 0.08
    seed: int = 0
    margin: float = 0.12  # grid inset from stage edges (stage units)


@dataclass(slots=True)
class Mole:
    id: int
    cell: int
    x: float
    y: float
    spawned_at: float
    expires_at: float


class WhackGame:
    def __init__(self, config: WhackConfig):
        self.cfg = config
        self.moles: dict[int, Mole] = {}
        self.score = 0
        self.phase = "playing"
        self._rng = random.Random(config.seed)
        self._next_mole_id = 1
        self._start: float | None = None
        self._next_spawn = 0.0
        self._last_marker: dict[int, tuple[float, float]] = {}

    def _cell_center(self, cell: int) -> tuple[float, float]:
        r, c = divmod(cell, self.cfg.cols)
        m = self.cfg.margin
        gx = m + (c + 0.5) * (1 - 2 * m) / self.cfg.cols
        gy = m + (r + 0.5) * (1 - 2 * m) / self.cfg.rows
        return (gx, gy)

    def _spawn(self, ts: float) -> None:
        occupied = {mo.cell for mo in self.moles.values()}
        free = [c for c in range(self.cfg.rows * self.cfg.cols) if c not in occupied]
        if not free:
            return
        cell = self._rng.choice(free)
        x, y = self._cell_center(cell)
        mo = Mole(
            id=self._next_mole_id,
            cell=cell,
            x=x,
            y=y,
            spawned_at=ts,
            expires_at=ts + self.cfg.mole_lifetime_s,
        )
        self.moles[mo.id] = mo
        self._next_mole_id += 1

    def tick(self, markers: list[tuple[int, float, float]], ts: float) -> list[Mole]:
        """markers: list of (marker_id, stage_x, stage_y). Returns active moles."""
        if self._start is None:
            self._start = ts
            self._next_spawn = ts
        if ts - self._start >= self.cfg.round_seconds:
            self.phase = "over"

        if self.phase == "playing":
            step = max(self.cfg.spawn_interval_s, 1e-3)
            while ts >= self._next_spawn:
                self._spawn(ts)
                self._next_spawn += step

        # prune stale marker history (markers absent this frame)
        present = {mid for mid, _, _ in markers}
        for mid in [m for m in self._last_marker if m not in present]:
            del self._last_marker[mid]

        # hits FIRST (so a hit at ts == expires_at still counts)
        for marker_id, x, y in markers:
            ax, ay = self._last_marker.get(marker_id, (x, y))
            self._last_marker[marker_id] = (x, y)
            for mid in list(self.moles):
                mo = self.moles[mid]
                if seg_point_dist(ax, ay, x, y, mo.x, mo.y) <= self.cfg.hit_radius:
                    self.score += self.cfg.points
                    del self.moles[mid]  # debounce: a mole scores once

        # expire AFTER hits
        for mid in [m for m, mo in self.moles.items() if ts >= mo.expires_at]:
            del self.moles[mid]

        return list(self.moles.values())

    def time_left_ms(self, ts: float) -> int:
        if self._start is None:
            return int(self.cfg.round_seconds * 1000)
        return max(0, int((self.cfg.round_seconds - (ts - self._start)) * 1000))
