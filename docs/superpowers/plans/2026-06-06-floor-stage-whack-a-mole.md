# Floor Stage Platform + Whack-a-Mole Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable floor-projection game platform (normalized stage bracketed by camera→stage and stage→projector homographies, ArUco markers, reactive sim) with Whack-a-Mole as its first game.

**Architecture:** local camera → ArUco markers → `StageCalibration` maps camera px → normalized stage (calibrated at playing height) → `WhackGame` (swept-trajectory hit test, schedule, scoring) → `SceneFrame` + `GameState` over WebSocket → renderer warps the stage onto the angled projector with a **perspective-correct** grid mesh. Reuses the reactive system's `SceneObject`/`SceneFrame` and the WS server.

**Tech Stack:** Python 3.11, numpy, opencv-contrib (cv2.aruco), websockets, p5.js (WebGL). Tests via pytest, headless.

**Spec:** `docs/superpowers/specs/2026-06-05-floor-stage-whack-a-mole-design.md`
**Branch:** `feat/floor-stage-whack` (stacked on `feat/reactive-objects`).

**Conventions:** ruff (E,F,I,B,UP; 100 col), type-hint public funcs, `logging` not `print`, numpy in hot paths, lazy-import cv2. Run changed files through `python -m ruff check`. Every commit ends with:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

**Correctness requirements (from spec review — build right the first time, do NOT defer):**
- Input homography is calibrated **at playing height** (constant-height marker plane → homography exact there); `height_offset` is only a residual.
- Output warp is **perspective-correct** (grid tessellation), NOT a two-triangle affine quad.

**Phases:**
- **A. Game core + transport** (Tasks 1–4) — stage math, GameState, WhackGame, floor source `step()`. All headless.
- **B. Real I/O** (Tasks 5–7) — ArUco detector, local camera, live loop + CLI.
- **C. Calibration + renderer** (Tasks 8–10) — persistence, perspective-correct renderer + calibration UI, end-to-end (at-height calibration, latency).

After Task 10 the rudimentary game is playable on the floor. Stop and report.

---

## Phase A — Game core + transport

### Task 1: `geometry/stage.py` — StageCalibration

**Files:**
- Create: `src/projectart/geometry/stage.py`
- Test: `tests/test_stage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage.py
from __future__ import annotations

import numpy as np
import pytest

from projectart.geometry.stage import STAGE_CORNERS, StageCalibration


def test_identity_maps_frame_to_unit_and_unit_to_projector():
    cal = StageCalibration.identity(frame_w=640, frame_h=360, proj_w=1920, proj_h=1080)
    assert cal.cam_px_to_stage(320, 180) == pytest.approx((0.5, 0.5))
    assert cal.stage_to_proj_px(0.5, 0.5) == pytest.approx((960, 540))


def test_from_corners_maps_corners_exactly():
    # camera sees the stage corners at these pixels (TL,TR,BR,BL):
    cam = [(100, 80), (540, 90), (560, 300), (90, 290)]
    proj = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
    cal = StageCalibration.from_corners(cam, STAGE_CORNERS, proj)
    for (px, py), (sx, sy) in zip(cam, STAGE_CORNERS):
        gx, gy = cal.cam_px_to_stage(px, py)
        assert (gx, gy) == pytest.approx((sx, sy), abs=1e-6)


def test_edge_midpoint_maps_where_predicted_through_composition():
    # A pure scaling homography: cam px = stage*1000. Midpoint of top edge
    # (stage 0.5,0) must come back to stage (0.5,0) — guards affine mistakes.
    cam = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    proj = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
    cal = StageCalibration.from_corners(cam, STAGE_CORNERS, proj)
    assert cal.cam_px_to_stage(500, 0) == pytest.approx((0.5, 0.0), abs=1e-6)
    assert cal.cam_px_to_stage(1000, 500) == pytest.approx((1.0, 0.5), abs=1e-6)


def test_height_offset_is_added_residual():
    cal = StageCalibration.identity(640, 360, 1920, 1080)
    cal.height_offset = (0.02, -0.01)
    gx, gy = cal.cam_px_to_stage(320, 180)
    assert (gx, gy) == pytest.approx((0.52, 0.49))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.geometry.stage`

- [ ] **Step 3: Write minimal implementation**

```python
# src/projectart/geometry/stage.py
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
    cam_to_stage: np.ndarray            # 3x3: camera px -> stage [0,1]
    stage_to_projector: np.ndarray      # 3x3: stage [0,1] -> projector px
    height_offset: tuple[float, float] = (0.0, 0.0)  # residual, stage units

    @classmethod
    def identity(cls, frame_w: int, frame_h: int, proj_w: int, proj_h: int) -> "StageCalibration":
        c2s = np.array([[1.0 / frame_w, 0, 0], [0, 1.0 / frame_h, 0], [0, 0, 1]], dtype=np.float64)
        s2p = np.array([[proj_w, 0, 0], [0, proj_h, 0], [0, 0, 1]], dtype=np.float64)
        return cls(cam_to_stage=c2s, stage_to_projector=s2p)

    @classmethod
    def from_corners(cls, cam_corners, stage_corners, proj_corners,
                     height_offset: tuple[float, float] = (0.0, 0.0)) -> "StageCalibration":
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stage.py -q && python -m ruff check src/projectart/geometry/stage.py tests/test_stage.py`
Expected: PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/projectart/geometry/stage.py tests/test_stage.py
git commit -m "feat(geometry): StageCalibration (cam<->stage<->projector homographies)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `server/protocol.py` — GameState + version bump

**Files:**
- Modify: `src/projectart/server/protocol.py`
- Test: `tests/test_protocol.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_protocol.py
from projectart.server.protocol import GameState, PROTOCOL_VERSION, to_dict


def test_protocol_version_is_3():
    assert PROTOCOL_VERSION == 3


def test_game_state_round_trips():
    d = to_dict(GameState(score=7, round_ms_left=42000, phase="playing", ts_ms=11))
    assert d["type"] == "game_state"
    assert d["score"] == 7
    assert d["round_ms_left"] == 42000
    assert d["phase"] == "playing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protocol.py -q`
Expected: FAIL — `ImportError: cannot import name 'GameState'` and version assert fails.

- [ ] **Step 3: Write implementation**

In `src/projectart/server/protocol.py`, change the version:

```python
PROTOCOL_VERSION = 3
```

Add after `SceneFrame`:

```python
@dataclass
class GameState:
    """Scoreboard / round state for a floor game. Sent alongside SceneFrame."""
    score: int = 0
    round_ms_left: int = 0
    phase: str = "playing"  # idle | playing | over
    ts_ms: int = 0
    type: Literal["game_state"] = "game_state"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_protocol.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/server/protocol.py tests/test_protocol.py
git commit -m "feat(protocol): GameState message + bump PROTOCOL_VERSION to 3" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `games/whack_a_mole.py` — game logic

**Files:**
- Create: `src/projectart/games/__init__.py`
- Create: `src/projectart/games/whack_a_mole.py`
- Test: `tests/test_whack_a_mole.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_whack_a_mole.py
from projectart.games.whack_a_mole import WhackConfig, WhackGame, seg_point_dist


def _cfg(**kw):
    base = dict(rows=3, cols=3, spawn_interval_s=1.0, mole_lifetime_s=10.0,
                points=1, round_seconds=100.0, hit_radius=0.08, seed=1, margin=0.1)
    base.update(kw)
    return WhackConfig(**base)


def test_spawn_cadence():
    g = WhackGame(_cfg())
    g.tick([], ts=0.0)          # spawns at t=0
    assert len(g.moles) == 1
    g.tick([], ts=0.5)          # before next interval
    assert len(g.moles) == 1
    g.tick([], ts=1.0)          # second spawn
    assert len(g.moles) == 2


def test_mole_expires():
    g = WhackGame(_cfg(mole_lifetime_s=1.0, spawn_interval_s=100.0))
    g.tick([], ts=0.0)
    assert len(g.moles) == 1
    g.tick([], ts=1.0)          # lifetime elapsed -> expired
    assert len(g.moles) == 0


def test_hit_scores_once_and_removes_mole():
    g = WhackGame(_cfg(spawn_interval_s=100.0))
    g.tick([], ts=0.0)
    mole = next(iter(g.moles.values()))
    # marker exactly on the mole
    g.tick([(1, mole.x, mole.y)], ts=0.1)
    assert g.score == 1
    assert mole.id not in g.moles
    # same marker next frame: no double count (mole gone)
    g.tick([(1, mole.x, mole.y)], ts=0.2)
    assert g.score == 1


def test_swept_segment_catches_fast_marker_that_jumps_over_mole():
    g = WhackGame(_cfg(spawn_interval_s=100.0, hit_radius=0.05))
    g.tick([], ts=0.0)
    mole = next(iter(g.moles.values()))
    # Two frames: marker is far on the left, then far on the right — both
    # single points miss, but the segment passes through the mole.
    g.tick([(1, mole.x - 0.3, mole.y)], ts=0.1)
    assert g.score == 0
    g.tick([(1, mole.x + 0.3, mole.y)], ts=0.2)
    assert g.score == 1


def test_miss_does_not_score():
    g = WhackGame(_cfg(spawn_interval_s=100.0, hit_radius=0.02))
    g.tick([], ts=0.0)
    mole = next(iter(g.moles.values()))
    g.tick([(1, mole.x + 0.5, mole.y + 0.5)], ts=0.1)
    assert g.score == 0


def test_round_over():
    g = WhackGame(_cfg(round_seconds=2.0))
    g.tick([], ts=0.0)
    assert g.phase == "playing"
    g.tick([], ts=2.0)
    assert g.phase == "over"


def test_seg_point_dist_endpoint_and_middle():
    assert seg_point_dist(0, 0, 10, 0, 5, 0) == 0.0
    assert seg_point_dist(0, 0, 10, 0, 5, 3) == 3.0
    assert seg_point_dist(0, 0, 0, 0, 3, 4) == 5.0  # degenerate segment = point
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_whack_a_mole.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.games.whack_a_mole`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/games/__init__.py
"""Floor games built on the stage platform + reactive sim."""
```

```python
# src/projectart/games/whack_a_mole.py
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
        mo = Mole(id=self._next_mole_id, cell=cell, x=x, y=y,
                  spawned_at=ts, expires_at=ts + self.cfg.mole_lifetime_s)
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
            while ts >= self._next_spawn:
                self._spawn(ts)
                self._next_spawn += self.cfg.spawn_interval_s

        for mid in [m for m, mo in self.moles.items() if ts >= mo.expires_at]:
            del self.moles[mid]

        for marker_id, x, y in markers:
            ax, ay = self._last_marker.get(marker_id, (x, y))
            self._last_marker[marker_id] = (x, y)
            for mid in list(self.moles):
                mo = self.moles[mid]
                if seg_point_dist(ax, ay, x, y, mo.x, mo.y) <= self.cfg.hit_radius:
                    self.score += self.cfg.points
                    del self.moles[mid]  # debounce: a mole scores once
        return list(self.moles.values())

    def time_left_ms(self, ts: float) -> int:
        if self._start is None:
            return int(self.cfg.round_seconds * 1000)
        return max(0, int((self.cfg.round_seconds - (ts - self._start)) * 1000))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_whack_a_mole.py -q && python -m ruff check src/projectart/games tests/test_whack_a_mole.py`
Expected: PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/projectart/games/__init__.py src/projectart/games/whack_a_mole.py tests/test_whack_a_mole.py
git commit -m "feat(games): whack-a-mole logic (schedule, swept-segment hit, scoring)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `inputs/floor_game.py` — FloorGameSource.step() + marker coasting

**Files:**
- Create: `src/projectart/inputs/floor_game.py`
- Test: `tests/test_floor_game.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_floor_game.py
from projectart.geometry.stage import StageCalibration
from projectart.games.whack_a_mole import WhackConfig, WhackGame
from projectart.inputs.floor_game import FloorGameSource, coast_markers


def _src():
    cal = StageCalibration.identity(frame_w=640, frame_h=360, proj_w=1920, proj_h=1080)
    game = WhackGame(WhackConfig(spawn_interval_s=100.0, seed=1))
    return FloorGameSource.for_testing(cal, game)


def test_step_emits_scene_and_gamestate_with_mole_in_stage_coords():
    src = _src()
    scene, gs = src.step([], ts=0.0)          # spawns one mole at t=0
    assert gs.type == "game_state"
    moles = [o for o in scene.objects if o.kind == "mole"]
    assert len(moles) == 1
    assert 0.0 <= moles[0].x <= 1.0 and 0.0 <= moles[0].y <= 1.0


def test_step_maps_marker_camera_px_to_stage_and_scores():
    src = _src()
    src.step([], ts=0.0)
    mole = next(iter(src.game.moles.values()))
    # camera px for that stage point under identity 640x360:
    cx, cy = mole.x * 640, mole.y * 360
    scene, gs = src.step([(1, cx, cy)], ts=0.1)
    assert gs.score == 1


def test_coast_markers_extrapolates_then_drops():
    state = {}
    # frame 1: marker seen at (0.2,0.2)
    out = coast_markers(state, [(1, 0.2, 0.2)], ts=0.0, max_coast_s=0.2)
    assert out == [(1, 0.2, 0.2)]
    # frame 2: seen at (0.4,0.2) -> velocity +0.2/s in x... dt 0.1 -> vx=2.0
    out = coast_markers(state, [(1, 0.4, 0.2)], ts=0.1, max_coast_s=0.2)
    assert out == [(1, 0.4, 0.2)]
    # frame 3: DROPOUT -> coast forward by vx*dt = 2.0*0.1 = 0.2 -> x~0.6
    out = coast_markers(state, [], ts=0.2, max_coast_s=0.2)
    assert len(out) == 1 and abs(out[0][1] - 0.6) < 1e-6
    # long gap beyond max_coast_s -> dropped
    out = coast_markers(state, [], ts=1.0, max_coast_s=0.2)
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_floor_game.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.inputs.floor_game`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/inputs/floor_game.py
"""Floor-game input source: local camera -> ArUco -> stage -> WhackGame ->
SceneFrame + GameState. `step()` is pure (no camera) for tests; `run()` adds the
live loop with marker coasting through brief detection dropouts.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..games.whack_a_mole import WhackGame
from ..geometry.stage import StageCalibration
from ..server.protocol import GameState, SceneFrame, SceneObject
from ..server.ws import Server

log = logging.getLogger(__name__)


def coast_markers(state: dict, detected: list[tuple[int, float, float]], ts: float,
                  max_coast_s: float = 0.15) -> list[tuple[int, float, float]]:
    """Carry markers through short dropouts. `state` is caller-owned and holds
    per-id (x, y, vx, vy, ts, seen_ts). Detected markers update velocity; missing
    ones are extrapolated until `max_coast_s` since last real detection, then dropped.
    Returns list of (id, stage_x, stage_y)."""
    out: list[tuple[int, float, float]] = []
    seen_ids = set()
    for mid, x, y in detected:
        seen_ids.add(mid)
        prev = state.get(mid)
        vx = vy = 0.0
        if prev is not None:
            dt = max(1e-3, ts - prev["ts"])
            vx, vy = (x - prev["x"]) / dt, (y - prev["y"]) / dt
        state[mid] = {"x": x, "y": y, "vx": vx, "vy": vy, "ts": ts, "seen_ts": ts}
        out.append((mid, x, y))
    for mid, s in list(state.items()):
        if mid in seen_ids:
            continue
        if ts - s["seen_ts"] > max_coast_s:
            del state[mid]
            continue
        dt = ts - s["ts"]
        nx, ny = s["x"] + s["vx"] * dt, s["y"] + s["vy"] * dt
        s["x"], s["y"], s["ts"] = nx, ny, ts
        out.append((mid, nx, ny))
    return out


class FloorGameSource:
    def __init__(self, server: Server | None, calib: StageCalibration, game: WhackGame,
                 camera_index: int = 0, frame_size: tuple[int, int] = (1280, 720),
                 target_hz: int = 60):
        self.server = server
        self.calib = calib
        self.game = game
        self.camera_index = camera_index
        self.frame_w, self.frame_h = frame_size
        self.target_hz = target_hz
        self._period = 1.0 / max(1, target_hz)
        self._coast: dict = {}
        self._capture = None
        self._detector = None

    @classmethod
    def for_testing(cls, calib: StageCalibration, game: WhackGame) -> "FloorGameSource":
        return cls(server=None, calib=calib, game=game)

    def step(self, markers_px: list[tuple[int, float, float]], ts: float) -> tuple[SceneFrame, GameState]:
        stage_markers = [(mid, *self.calib.cam_px_to_stage(cx, cy)) for mid, cx, cy in markers_px]
        moles = self.game.tick(stage_markers, ts)
        objs = [SceneObject(id=mo.id, kind="mole", shape="circle", x=mo.x, y=mo.y,
                            r=self.game.cfg.hit_radius, color="#cc3333", alpha=1.0)
                for mo in moles]
        ts_ms = int(ts * 1000)
        scene = SceneFrame(ts_ms=ts_ms, objects=objs)
        gs = GameState(score=self.game.score, round_ms_left=self.game.time_left_ms(ts),
                       phase=self.game.phase, ts_ms=ts_ms)
        return scene, gs

    async def run(self) -> None:
        from ..capture.local_cam import LocalCamera
        from ..detection.aruco import ArucoDetector
        assert self.server is not None
        self._capture = LocalCamera(index=self.camera_index, name="local")
        self._detector = ArucoDetector()
        self._capture.start()
        log.info("floor game starting (camera_index=%d, target_hz=%d)", self.camera_index, self.target_hz)
        loop_t0 = time.monotonic()
        try:
            while True:
                t0 = time.monotonic()
                frame = self._capture.latest()
                if frame is None:
                    await asyncio.sleep(self._period)
                    continue
                self.frame_h, self.frame_w = frame.image.shape[:2]
                ts = time.monotonic() - loop_t0
                detected = [(m.id, m.cx, m.cy) for m in self._detector(frame.image)]
                # coast in CAMERA px then map to stage in step()
                coasted = coast_markers(self._coast, detected, ts)
                scene, gs = self.step(coasted, ts)
                await self.server.broadcast(scene)
                await self.server.broadcast(gs)
                await asyncio.sleep(max(0.0, self._period - (time.monotonic() - t0)))
        finally:
            self._capture.stop()
```

> Note: `coast_markers` operates in whatever space it's fed; in `run()` we coast in **camera px** then map to stage in `step()`. The test feeds stage-like values directly — same function, space-agnostic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_floor_game.py -q && python -m ruff check src/projectart/inputs/floor_game.py tests/test_floor_game.py`
Expected: PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/projectart/inputs/floor_game.py tests/test_floor_game.py
git commit -m "feat(inputs): FloorGameSource step() + marker coasting" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase B — Real I/O

### Task 5: `detection/aruco.py` — ArUco marker detector

**Files:**
- Create: `src/projectart/detection/aruco.py`
- Test: `tests/test_aruco.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aruco.py
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _marker_image(marker_id: int, size: int = 200, border: int = 60):
    import cv2 as _cv2
    d = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_4X4_50)
    img = _cv2.aruco.generateImageMarker(d, marker_id, size)
    canvas = np.full((size + 2 * border, size + 2 * border), 255, dtype=np.uint8)
    canvas[border:border + size, border:border + size] = img
    return _cv2.cvtColor(canvas, _cv2.COLOR_GRAY2BGR)


def test_detects_marker_id_and_center():
    from projectart.detection.aruco import ArucoDetector
    img = _marker_image(7)
    markers = ArucoDetector()(img)
    assert len(markers) == 1
    assert markers[0].id == 7
    h, w = img.shape[:2]
    assert markers[0].cx == pytest.approx(w / 2, abs=3)
    assert markers[0].cy == pytest.approx(h / 2, abs=3)


def test_no_markers_returns_empty():
    from projectart.detection.aruco import ArucoDetector
    blank = np.full((240, 320, 3), 127, dtype=np.uint8)
    assert ArucoDetector()(blank) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aruco.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.detection.aruco`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/detection/aruco.py
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
    corners: np.ndarray   # (4,2) float, camera px
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
        for c, i in zip(corners, ids.flatten()):
            pts = np.asarray(c, dtype=np.float64).reshape(4, 2)
            out.append(Marker(id=int(i), corners=pts,
                              cx=float(pts[:, 0].mean()), cy=float(pts[:, 1].mean())))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_aruco.py -q && python -m ruff check src/projectart/detection/aruco.py tests/test_aruco.py`
Expected: PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/projectart/detection/aruco.py tests/test_aruco.py
git commit -m "feat(detection): ArUco marker detector (DICT_4X4_50, stable ids)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `capture/local_cam.py` — local/iPhone webcam capture

**Files:**
- Create: `src/projectart/capture/local_cam.py`

**Verification is manual** (needs a camera). No pytest. Mirrors `YiCapture`'s thread + Q=1 interface and reuses its `Frame`.

- [ ] **Step 1: Write implementation**

```python
# src/projectart/capture/local_cam.py
"""Low-latency local webcam capture (Continuity Camera / USB).

Mirrors YiCapture: one daemon thread, depth-1 newest-wins queue, `latest()`
returns the most recent frame or None. AVFoundation backend on macOS. cv2 is
imported lazily.

IMPORTANT (Continuity Camera): disable auto-effects, ESPECIALLY Center Stage —
its per-frame reframing breaks the fixed camera->stage homography. Use a short
shutter with enough light to avoid motion blur on fast swings.
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

from .yi_rtsp import Frame  # reuse the (image, ts_ms) dataclass

log = logging.getLogger(__name__)


class LocalCamera:
    def __init__(self, index: int = 0, name: str = "local", width: int = 1280, height: int = 720):
        self.index = index
        self.name = name
        self.width = width
        self.height = height
        self._q: Queue[Frame] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"localcam[{self.name}]", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self) -> Optional[Frame]:
        try:
            return self._q.get_nowait()
        except Empty:
            return None

    def _put_newest(self, frame: Frame) -> None:
        try:
            self._q.get_nowait()
        except Empty:
            pass
        try:
            self._q.put_nowait(frame)
        except Exception:
            pass

    def _loop(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            pass
        if not cap.isOpened():
            log.error("[%s] could not open camera index %d", self.name, self.index)
            cap.release()
            return
        log.info("[%s] local camera open (index=%d)", self.name, self.index)
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok or img is None:
                time.sleep(0.005)
                continue
            self._put_newest(Frame(image=img, ts_ms=int(time.monotonic() * 1000)))
        cap.release()
```

- [ ] **Step 2: Smoke check (manual / non-fatal)**

Run: `python -c "from projectart.capture.local_cam import LocalCamera; print('import ok')"`
Expected: `import ok` (live capture verified in Task 10).

- [ ] **Step 3: Commit**

```bash
git add src/projectart/capture/local_cam.py
git commit -m "feat(capture): local/Continuity-Camera webcam capture (Q=1, AVFoundation)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: CLI wiring — `--input floor --game whack`

**Files:**
- Modify: `src/projectart/__main__.py`
- Modify: `src/projectart/app.py`
- Create: `src/projectart/inputs/floor_factory.py`

- [ ] **Step 1: Add CLI flags**

In `src/projectart/__main__.py`, add `floor` to choices and a `--game` flag:

```python
        choices=["mouse", "gloves", "scene", "reactive", "floor", "wand", "androidtv"],
```

```python
    p.add_argument("--game", default="whack", help="floor game to run (default: whack)")
    p.add_argument("--camera-index", type=int, default=0, help="local camera index for --input floor")
```

- [ ] **Step 2: Add a factory that builds the source**

```python
# src/projectart/inputs/floor_factory.py
"""Build a FloorGameSource from CLI args + persisted calibration."""
from __future__ import annotations

import logging

from ..calibration.persist import load_calibration, stage_calibration_from_doc
from ..games.whack_a_mole import WhackConfig, WhackGame
from ..geometry.stage import StageCalibration
from ..inputs.floor_game import FloorGameSource
from ..server.ws import Server

log = logging.getLogger(__name__)


def build_floor_source(canvas_size: tuple[int, int], server: Server, game: str,
                       camera_index: int) -> FloorGameSource:
    if game != "whack":
        log.warning("unknown game %r; defaulting to whack", game)
    calib = None
    doc = load_calibration()
    if doc is not None:
        calib = stage_calibration_from_doc(doc)
    if calib is None:
        log.warning("no stage calibration found; using identity (uncalibrated — wrong but visible)")
        calib = StageCalibration.identity(frame_w=1280, frame_h=720,
                                          proj_w=canvas_size[0], proj_h=canvas_size[1])
    return FloorGameSource(server=server, calib=calib, game=WhackGame(WhackConfig()),
                           camera_index=camera_index)
```

- [ ] **Step 3: Wire into `app.py`**

In `src/projectart/app.py`, add after the `reactive` branch:

```python
        elif self.args.input == "floor":
            from .inputs.floor_factory import build_floor_source

            source = build_floor_source(
                canvas_size=self.canvas_size,
                server=self._server,
                game=getattr(self.args, "game", "whack"),
                camera_index=getattr(self.args, "camera_index", 0),
            )
            await source.run()
```

- [ ] **Step 4: Verify CLI parses (calibration helper lands in Task 8; until then this import will fail — so gate this step after Task 8, or stub).**

> Dependency note: `stage_calibration_from_doc` is implemented in Task 8. Implement Task 8 before running this step. After Task 8:

Run: `python -m projectart --input floor --help >/dev/null && python -m pytest -q`
Expected: help exits 0; suite passes.

- [ ] **Step 5: Commit**

```bash
git add src/projectart/__main__.py src/projectart/app.py src/projectart/inputs/floor_factory.py
git commit -m "feat(cli): --input floor --game whack + floor source factory" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase C — Calibration + renderer

### Task 8: `calibration/persist.py` — stage calibration persistence

**Files:**
- Modify: `src/projectart/calibration/persist.py`
- Test: `tests/test_calibration_persist.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_calibration_persist.py
import numpy as np

from projectart.calibration.persist import (
    CalibrationDoc, StageSchema, stage_calibration_from_doc, doc_with_stage,
)
from projectart.geometry.stage import StageCalibration


def test_stage_round_trips_through_doc():
    cal = StageCalibration.identity(640, 360, 1920, 1080)
    cal.height_offset = (0.01, -0.02)
    doc = doc_with_stage(CalibrationDoc(), cal)
    back = stage_calibration_from_doc(doc)
    assert back is not None
    assert np.allclose(back.cam_to_stage, cal.cam_to_stage)
    assert np.allclose(back.stage_to_projector, cal.stage_to_projector)
    assert back.height_offset == (0.01, -0.02)


def test_no_stage_section_returns_none():
    assert stage_calibration_from_doc(CalibrationDoc()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration_persist.py -q`
Expected: FAIL — `ImportError: cannot import name 'StageSchema'`

- [ ] **Step 3: Write implementation**

In `src/projectart/calibration/persist.py`, add the schema (near the other schemas):

```python
class StageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cam_to_stage: list[list[float]] = Field(min_length=3, max_length=3)
    stage_to_projector: list[list[float]] = Field(min_length=3, max_length=3)
    height_offset: list[float] = Field(default=[0.0, 0.0], min_length=2, max_length=2)
```

Add the field to `CalibrationDoc`:

```python
    stage: Optional[StageSchema] = None
```

Add the converters at the end of the file:

```python
def stage_calibration_from_doc(doc: "CalibrationDoc"):
    """Build a StageCalibration from a doc's stage section, or None if absent."""
    import numpy as np

    from ..geometry.stage import StageCalibration

    if doc.stage is None:
        return None
    return StageCalibration(
        cam_to_stage=np.asarray(doc.stage.cam_to_stage, dtype=np.float64),
        stage_to_projector=np.asarray(doc.stage.stage_to_projector, dtype=np.float64),
        height_offset=(float(doc.stage.height_offset[0]), float(doc.stage.height_offset[1])),
    )


def doc_with_stage(doc: "CalibrationDoc", cal) -> "CalibrationDoc":
    """Return a copy of `doc` with its stage section set from a StageCalibration."""
    return doc.model_copy(update={"stage": StageSchema(
        cam_to_stage=[list(map(float, row)) for row in cal.cam_to_stage],
        stage_to_projector=[list(map(float, row)) for row in cal.stage_to_projector],
        height_offset=[float(cal.height_offset[0]), float(cal.height_offset[1])],
    )})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration_persist.py -q && python -m ruff check src/projectart/calibration/persist.py`
Expected: PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/projectart/calibration/persist.py tests/test_calibration_persist.py
git commit -m "feat(calibration): persist stage calibration (homographies + height offset)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Renderer — perspective-correct floor warp + HUD + calibration UI

**Files:**
- Create: `renderer/floor.html`
- Create: `renderer/floor.js`
- Modify: `src/projectart/inputs/floor_game.py` (inbound calibration message handler)

**Verification is manual** (browser + projector, in Task 10). The warp MUST be perspective-correct (grid mesh).

- [ ] **Step 1: Write the renderer page**

```html
<!-- renderer/floor.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ProjectArt — Floor Game</title>
  <style>
    html, body { margin: 0; height: 100%; background: #000; overflow: hidden; cursor: crosshair; }
    canvas { display: block; }
    #status { position: fixed; top: 8px; left: 8px; color: rgba(255,255,255,.5);
      font: 12px ui-monospace, Menlo, monospace; pointer-events: none; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>
</head>
<body>
  <div id="status">connecting…  [c]=calibrate corners  [1-4]=set input corner</div>
  <script src="ws_client.js"></script>
  <script src="floor.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write the renderer (perspective-correct grid warp)**

```javascript
// renderer/floor.js
// Renders the stage (moles + HUD) to an offscreen buffer in stage space, then
// warps it onto the floor through a 4-corner homography using a FINE GRID MESH
// (perspective-correct — a two-triangle quad would bow interior lines). The 4
// projector corners are draggable (output calibration) and persisted upstream.
(function () {
  const GRID = 24;                 // mesh resolution (perspective-correct warp)
  const STAGE_W = 1000, STAGE_H = 1000;
  let stageBuf;
  let moles = new Map();           // id -> {x,y,r,color}
  let game = { score: 0, round_ms_left: 0, phase: 'playing' };
  let calibrate = false;
  // output corners (projector px), stage order TL,TR,BR,BL; default = full screen
  let corners = null;
  let dragging = -1;

  function loadCorners() {
    try { const s = localStorage.getItem('PA_FLOOR_CORNERS'); if (s) return JSON.parse(s); } catch (e) {}
    return [[0, 0], [windowWidth, 0], [windowWidth, windowHeight], [0, windowHeight]];
  }
  function saveCorners() {
    try { localStorage.setItem('PA_FLOOR_CORNERS', JSON.stringify(corners)); } catch (e) {}
    window.PA_WS.send({ type: 'calib_output', corners });
  }

  // 4-point homography (unit square stage [0,1] -> dst px). Standard solution.
  function homography(dst) {
    const [p0, p1, p2, p3] = dst; // TL,TR,BR,BL == (0,0),(1,0),(1,1),(0,1)
    const x0 = p0[0], y0 = p0[1], x1 = p1[0], y1 = p1[1];
    const x2 = p2[0], y2 = p2[1], x3 = p3[0], y3 = p3[1];
    const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
    const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
    const den = dx1 * dy2 - dx2 * dy1;
    const g = (dx3 * dy2 - dx2 * dy3) / den;
    const h = (dx1 * dy3 - dx3 * dy1) / den;
    return [
      [x1 - x0 + g * x1, x3 - x0 + h * x3, x0],
      [y1 - y0 + g * y1, y3 - y0 + h * y3, y0],
      [g, h, 1],
    ];
  }
  function applyH(H, u, v) {
    const x = H[0][0] * u + H[0][1] * v + H[0][2];
    const y = H[1][0] * u + H[1][1] * v + H[1][2];
    const w = H[2][0] * u + H[2][1] * v + H[2][2];
    return [x / w, y / w];
  }

  function onMessage(msg) {
    if (msg.type === 'scene_frame') {
      const seen = new Set();
      for (const o of msg.objects) { if (o.kind === 'mole') { moles.set(o.id, o); seen.add(o.id); } }
      for (const id of [...moles.keys()]) if (!seen.has(id)) moles.delete(id);
    } else if (msg.type === 'game_state') {
      game = msg;
    }
  }

  window.setup = function () {
    createCanvas(windowWidth, windowHeight, WEBGL);
    stageBuf = createGraphics(STAGE_W, STAGE_H);
    corners = loadCorners();
    window.PA_WS.on('message', onMessage);
  };
  window.windowResized = function () { resizeCanvas(windowWidth, windowHeight); };

  function drawStage() {
    stageBuf.push();
    stageBuf.background(0);
    // moles
    stageBuf.noStroke();
    for (const o of moles.values()) {
      stageBuf.fill(o.color || '#cc3333');
      stageBuf.circle(o.x * STAGE_W, o.y * STAGE_H, (o.r || 0.08) * 2 * STAGE_W);
    }
    // HUD on the stage (so it projects on the floor)
    stageBuf.fill(255);
    stageBuf.textSize(40);
    stageBuf.text('Score ' + game.score, 30, 50);
    stageBuf.text((game.round_ms_left / 1000 | 0) + 's', STAGE_W - 140, 50);
    if (game.phase === 'over') { stageBuf.textSize(90); stageBuf.text('TIME!', STAGE_W / 2 - 130, STAGE_H / 2); }
    stageBuf.pop();
  }

  window.draw = function () {
    drawStage();
    background(0);
    const H = homography(corners);
    // perspective-correct warp: NxN grid, each vertex placed via the homography
    texture(stageBuf);
    noStroke();
    translate(-width / 2, -height / 2);  // WEBGL origin -> top-left pixels
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const u0 = i / GRID, u1 = (i + 1) / GRID, v0 = j / GRID, v1 = (j + 1) / GRID;
        const a = applyH(H, u0, v0), b = applyH(H, u1, v0);
        const c = applyH(H, u1, v1), d = applyH(H, u0, v1);
        beginShape();
        vertex(a[0], a[1], 0, u0 * STAGE_W, v0 * STAGE_H);
        vertex(b[0], b[1], 0, u1 * STAGE_W, v0 * STAGE_H);
        vertex(c[0], c[1], 0, u1 * STAGE_W, v1 * STAGE_H);
        vertex(d[0], d[1], 0, u0 * STAGE_W, v1 * STAGE_H);
        endShape(CLOSE);
      }
    }
    if (calibrate) drawHandles();
  };

  function drawHandles() {
    noFill(); stroke(0, 255, 0); strokeWeight(2);
    beginShape();
    for (const c of corners) vertex(c[0], c[1]);
    endShape(CLOSE);
    fill(0, 255, 0); noStroke();
    const labels = ['TL', 'TR', 'BR', 'BL'];
    for (let i = 0; i < 4; i++) { circle(corners[i][0], corners[i][1], 18); }
  }

  window.keyPressed = function () {
    if (key === 'c' || key === 'C') calibrate = !calibrate;
    if (['1', '2', '3', '4'].includes(key)) {
      // request backend capture the current marker at this stage corner (input calib)
      window.PA_WS.send({ type: 'calib_input_capture', corner: parseInt(key, 10) - 1 });
    }
  };
  window.mousePressed = function () {
    if (!calibrate) return;
    for (let i = 0; i < 4; i++) {
      if (dist(mouseX, mouseY, corners[i][0], corners[i][1]) < 16) { dragging = i; return; }
    }
  };
  window.mouseDragged = function () { if (dragging >= 0) corners[dragging] = [mouseX, mouseY]; };
  window.mouseReleased = function () { if (dragging >= 0) { dragging = -1; saveCorners(); } };
})();
```

- [ ] **Step 3: Add the inbound calibration handler to the floor source**

In `src/projectart/inputs/floor_game.py`, in `run()` after creating the detector, register a handler so the renderer can drive input calibration and persist output corners. Add this method to `FloorGameSource`:

```python
    def _on_client_message(self, _ws, msg: dict):
        import asyncio as _asyncio
        from ..calibration.persist import load_calibration, doc_with_stage, save_calibration

        async def _noop():
            return None

        mtype = msg.get("type")
        if mtype == "calib_output":
            # corners: 4 projector px in stage order TL,TR,BR,BL
            from ..geometry.stage import STAGE_CORNERS, _homography
            try:
                self.calib.stage_to_projector = _homography(STAGE_CORNERS, msg["corners"])
                doc = load_calibration() or __import__("projectart.calibration.persist",
                      fromlist=["CalibrationDoc"]).CalibrationDoc()
                save_calibration(doc_with_stage(doc, self.calib))
                log.info("output calibration updated + saved")
            except Exception:
                log.exception("bad calib_output message")
        elif mtype == "calib_input_capture":
            # associate the latest single marker's camera px with stage corner i
            corner = int(msg.get("corner", 0))
            self._pending_input_corner = corner
            log.info("will capture next marker for stage corner %d", corner)
        return _noop()
```

And in `run()`, register it and implement input-corner capture in the loop. After `self._detector = ArucoDetector()` add:

```python
        self._pending_input_corner = None
        self._input_cam_corners = [None, None, None, None]
        if self.server is not None:
            self.server.set_message_handler(self._on_client_message)
```

In the loop, right after computing `detected`, add capture handling:

```python
                if self._pending_input_corner is not None and detected:
                    self._input_cam_corners[self._pending_input_corner] = (detected[0][1], detected[0][2])
                    self._pending_input_corner = None
                    if all(c is not None for c in self._input_cam_corners):
                        from ..geometry.stage import STAGE_CORNERS, StageCalibration
                        from ..calibration.persist import load_calibration, doc_with_stage, save_calibration
                        self.calib = StageCalibration.from_corners(
                            self._input_cam_corners, STAGE_CORNERS,
                            # keep current output by reading projector corners back is non-trivial;
                            # reuse existing stage_to_projector via its mapping of STAGE_CORNERS:
                            [self.calib.stage_to_proj_px(*c) for c in STAGE_CORNERS],
                            height_offset=self.calib.height_offset)
                        doc = load_calibration()
                        if doc is None:
                            from ..calibration.persist import CalibrationDoc
                            doc = CalibrationDoc()
                        save_calibration(doc_with_stage(doc, self.calib))
                        log.info("input calibration (camera->stage) updated + saved")
```

> This keeps `stage_to_projector` intact while replacing `cam_to_stage` from the four captured camera positions (markers placed at playing height on each projected corner — see Task 10).

- [ ] **Step 4: Static check + ruff**

Run: `python -c "import pathlib; assert pathlib.Path('renderer/floor.js').exists() and pathlib.Path('renderer/floor.html').exists()" && python -m ruff check src/projectart/inputs/floor_game.py && python -m pytest tests/test_floor_game.py -q`
Expected: no error; ruff clean; floor tests still pass.

- [ ] **Step 5: Commit**

```bash
git add renderer/floor.html renderer/floor.js src/projectart/inputs/floor_game.py
git commit -m "feat(renderer): perspective-correct floor warp + HUD + calibration UI" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: End-to-end — at-height calibration, latency, play

**Files:**
- Modify: `README.md` (a short "Floor games" run section)

**Manual**, with the projector + iPhone. This is the rudimentary checkpoint.

- [ ] **Step 1: Prep the camera**
On the Mac: enable Continuity Camera, then **turn OFF Center Stage** and other video effects (Control Center → Video Effects while the camera is in use). Ensure good lighting / short shutter.

- [ ] **Step 2: Launch**

```bash
python -m projectart --input floor --game whack --camera-index 0 --http-port 8011 --ws-port 8766
# open: http://127.0.0.1:8011/floor.html?ws=ws://127.0.0.1:8766/  on the projector display
```

- [ ] **Step 3: Output calibration (geometry of the angled projector)**
Press `c`; drag the 4 green corners until the projected stage is a clean rectangle on the floor; release (auto-saves). Confirm a straight line across the stage stays straight (perspective-correct warp — AC-1).

- [ ] **Step 4: Input calibration AT PLAYING HEIGHT (AC-2)**
Place the ArUco marker on a block of the mallet/foot's playing height, set it on the **TL** projected corner, press `1`; repeat on TR/BR/BL with `2`/`3`/`4`. After the 4th, `cam_to_stage` is computed + saved. Move the marker around — moles should register where the marker visually is, across the whole stage.

- [ ] **Step 5: Measure latency + play (AC-8)**
Temporarily log per-loop timing in `run()` (or observe swing→hit responsiveness). Confirm a real swing registers (swept-segment + coast). Note end-to-end feel (target ≈ ≤100 ms).

- [ ] **Step 6: Full suite + README + commit**

Run: `python -m pytest -q`
Expected: all pass.

Add a short "Floor games (whack-a-mole)" section to `README.md` with the launch + calibration steps above, then:

```bash
git add README.md
git commit -m "docs: floor games run + calibration instructions" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Stop and report to the user.** Rudimentary whack-a-mole runs on the floor. Pause for feedback / the car-control follow-on spec.

---

## Self-Review

**Spec coverage:**
- §1 stage + two homographies + height offset → Task 1 (`StageCalibration`); calibration mode → Task 9 (UI) + Task 10 (procedure); persistence → Task 8. ✔
- §2 ArUco (`DICT_4X4_50`) → Task 5; local/Continuity camera + Center-Stage-off → Task 6 + Task 10 step 1. ✔
- §3 whack logic (swept-segment, debounce, schedule, aspect note) → Task 3; mole/hit radius in stage units used in Task 3/4. ✔
- §4 renderer perspective-correct grid warp + HUD + GameState → Task 9 + Task 2. ✔
- §5 module layout / pipeline → Tasks 1–9; `--input floor --game whack` → Task 7. ✔
- §6 testing (stage round-trip + edge-midpoint, ArUco synthetic, whack, GameState) → Tasks 1,3,5,2 tests; coast tested in Task 4. ✔
- §7 acceptance AC-1..AC-8 → AC-1/AC-2/AC-8 verified in Task 10; AC-3 Task 5; AC-4 Task 3; AC-5 Tasks 8/9; AC-6 WhackConfig (Task 3); AC-7 full suite (Task 10 step 6). ✔
- §8 future (car loop) → out of scope, noted. ✔

**Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output. Task 7 step 4 explicitly notes its dependency on Task 8 (ordering), not a placeholder.

**Type consistency:** `StageCalibration.cam_px_to_stage/stage_to_proj_px/from_corners/identity` + `STAGE_CORNERS` + `_homography`; `GameState(score, round_ms_left, phase, ts_ms)`; `SceneObject`/`SceneFrame` (reactive); `WhackConfig`/`Mole`/`WhackGame.tick(markers, ts)`/`.moles`/`.score`/`.phase`/`.time_left_ms`; `seg_point_dist`; `FloorGameSource.step/for_testing/run` + `coast_markers(state, detected, ts, max_coast_s)`; `Marker(id, corners, cx, cy)`/`ArucoDetector`; `LocalCamera`/`Frame`; `StageSchema`/`stage_calibration_from_doc`/`doc_with_stage`; `build_floor_source`. Names consistent across tasks.

**Ordering note:** Task 7 imports `stage_calibration_from_doc` (Task 8) — implement Task 8 before Task 7's step 4 verification (flagged inline). All other tasks are in dependency order.
