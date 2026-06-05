# Reactive Objects & Stable Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn camera detections into persistent, ID-tracked reactive objects that move/morph/collide according to one JSON config tree, streamed to a mock renderer that morphs one box per tracked ID instead of recreating it.

**Architecture:** ultralytics persistent tracker → `TrackedRegistry` (keyed on track_id + confirm/coast hysteresis) → backend reference `Simulator` (rules → behaviors → physics) → `SceneFrame` snapshots over WebSocket → p5 mock that tweens visuals keyed by object id. Normalized world space `[0,1]`. Backend is authoritative; the config tree is the game-engine-portable contract.

**Tech Stack:** Python 3.11, numpy, ultralytics (detection module only), pydantic-free dataclasses for config, websockets, p5.js renderer. Tests via pytest, all headless.

**Spec:** `docs/superpowers/specs/2026-06-05-reactive-objects-design.md`

**Conventions:** ruff (100 col), type-hint public funcs, `logging` not `print`, numpy in hot paths, lazy-import cv2/ultralytics. Every commit ends with the trailer:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

**Phases:**
- **A. Tracking foundations** (Tasks 1–4) — mapping, entity velocity, registry hysteresis, detector track ids.
- **B. Reactive core** (Tasks 5–10) — config, objects, physics, behaviors, rules, simulator.
- **C. Wire + pipeline + mock** (Tasks 11–15) — SceneFrame, ReactiveSource, renderer, CLI, end-to-end.

After Task 15 the rudimentary demo runs. Stop and report to the user (projector integration is next, mocked via the identity transform until then).

---

## Phase A — Tracking foundations

### Task 1: `geometry/mapping.py` — cam-pixel → normalized world transform

**Files:**
- Create: `src/projectart/geometry/mapping.py`
- Test: `tests/test_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mapping.py
from __future__ import annotations

import numpy as np
import pytest

from projectart.geometry.mapping import CamToWorld


def test_identity_divides_by_frame_size():
    m = CamToWorld.identity()
    assert m(320, 180, 640, 360) == pytest.approx((0.5, 0.5))
    assert m(0, 0, 640, 360) == pytest.approx((0.0, 0.0))
    assert m(640, 360, 640, 360) == pytest.approx((1.0, 1.0))


def test_from_config_identity_string():
    assert CamToWorld.from_config("identity").matrix is None
    assert CamToWorld.from_config(None).matrix is None


def test_homography_maps_pixels_to_world():
    # A homography that scales pixels by 1/1000 (so 500px -> 0.5 world).
    H = [[1 / 1000, 0, 0], [0, 1 / 1000, 0], [0, 0, 1]]
    m = CamToWorld.from_config(H)
    # frame size is ignored when a matrix is present
    assert m(500, 250, 640, 360) == pytest.approx((0.5, 0.25))


def test_bad_homography_shape_raises():
    with pytest.raises(ValueError):
        CamToWorld.from_config([[1, 0], [0, 1]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mapping.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.geometry.mapping`

- [ ] **Step 3: Write minimal implementation**

```python
# src/projectart/geometry/mapping.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mapping.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/projectart/geometry/mapping.py tests/test_mapping.py
git commit -m "feat(geometry): CamToWorld cam-pixel to normalized world transform" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `tracking/entity.py` — smoothed velocity + hits/confirmed

**Files:**
- Modify: `src/projectart/tracking/entity.py`
- Test: `tests/test_tracking_velocity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_velocity.py
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.tracking.entity import TrackedEntity


def _det(cx, cy):
    return Detection(class_id=0, cx=cx, cy=cy, w=20, h=20, confidence=0.9, class_name="person")


def test_new_entity_zero_velocity_and_one_hit():
    e = TrackedEntity(track_id=1, det=_det(100, 100), ts=0.0)
    assert e.velocity == (0.0, 0.0)
    assert e.hits == 1
    assert e.confirmed is False
    assert e.center == (100.0, 100.0)


def test_constant_velocity_converges():
    # Move +10px/frame in x at 10 fps -> ~100 px/s. 1€ smoothing converges.
    e = TrackedEntity(track_id=1, det=_det(100, 100), ts=0.0)
    x = 100.0
    for i in range(1, 12):
        x += 10.0
        e.on_update(_det(x, 100), ts=i * 0.1)
    vx, vy = e.velocity
    assert vx == __import__("pytest").approx(100.0, rel=0.2)
    assert abs(vy) < 5.0
    assert e.hits == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking_velocity.py -q`
Expected: FAIL — `AttributeError: 'TrackedEntity' object has no attribute 'velocity'`

- [ ] **Step 3: Write minimal implementation**

Add the import near the top of `src/projectart/tracking/entity.py` (after the existing imports):

```python
import numpy as np

from ..geometry.filter_one_euro import OneEuroFilter
```

In `TrackedEntity.__init__`, after `self.last_seen_ts = now`, add:

```python
        # Smoothed centre (cam px) + finite-difference velocity (cam px/s).
        self.center: tuple[float, float] = (float(det.cx), float(det.cy))
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.hits: int = 1
        self.confirmed: bool = False
        self._pos_filter = OneEuroFilter(mincutoff=1.0, beta=0.05)
        self._pos_filter(np.array(self.center, dtype=np.float64), t=now)
        self._vel_ts: float = now
```

Replace the body of `on_update` with:

```python
    def on_update(self, det: Detection, ts: float) -> None:
        """Called every frame this entity is detected (after the first).
        Updates bbox, smoothed centre, finite-difference velocity, hit count."""
        self.last_bbox = BBox.from_detection(det)
        self.last_confidence = float(det.confidence)
        self.last_seen_ts = ts
        self.hits += 1
        sm = self._pos_filter(np.array([det.cx, det.cy], dtype=np.float64), t=ts)
        dt = max(1e-3, ts - self._vel_ts)
        self.velocity = (
            float((sm[0] - self.center[0]) / dt),
            float((sm[1] - self.center[1]) / dt),
        )
        self.center = (float(sm[0]), float(sm[1]))
        self._vel_ts = ts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_velocity.py tests/test_tracking.py -q`
Expected: PASS (existing tracking tests still green; on_enter still fires via registry which we change next)

- [ ] **Step 5: Commit**

```bash
git add src/projectart/tracking/entity.py tests/test_tracking_velocity.py
git commit -m "feat(tracking): smoothed velocity + hits/confirmed on TrackedEntity" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `tracking/registry.py` — track_id keying + confirm/coast hysteresis

**Files:**
- Modify: `src/projectart/tracking/registry.py`
- Modify: `src/projectart/tracking/entity.py` (add `track_key`)
- Test: `tests/test_tracking_hysteresis.py`

**Design:** Default behaviour unchanged (greedy-IoU, immediate enter) so existing tests pass. New params opt in: `confirm_after_hits` (default 1), `lost_after_s`/`gone_after_s` (default None → use entity class attrs), `min_confidence`. When detections carry `track_id`, associate by id instead of IoU. `on_enter` fires only when an entity becomes confirmed (`hits >= confirm_after_hits`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_hysteresis.py
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.tracking import TrackedRegistry
from projectart.tracking.builtins import Person


def _det(tid, cx=100.0, cy=100.0, conf=0.9):
    return Detection(class_id=0, cx=cx, cy=cy, w=20, h=20, confidence=conf,
                     class_name="person", track_id=tid)


def test_track_id_association_keeps_one_entity_when_bbox_jumps():
    reg = TrackedRegistry(entity_types=[Person])
    reg.consume([_det(7, cx=100)], ts=0.0)
    ent = next(iter(reg))
    # Large jump that greedy-IoU would treat as a new object — track_id keeps it.
    reg.consume([_det(7, cx=600)], ts=0.05)
    assert len(reg) == 1
    assert next(iter(reg)).track_id == ent.track_id


def test_confirm_after_hits_suppresses_one_frame_flicker():
    enters = []
    reg = TrackedRegistry(entity_types=[Person], confirm_after_hits=3)

    class WiredPerson(Person):
        def on_enter(self):
            enters.append(self.track_id)

    reg = TrackedRegistry(entity_types=[WiredPerson], confirm_after_hits=3)
    reg.consume([_det(1)], ts=0.0)      # hit 1 -> not confirmed
    assert enters == []
    assert next(iter(reg)).confirmed is False
    reg.consume([_det(1)], ts=0.03)     # hit 2
    assert enters == []
    reg.consume([_det(1)], ts=0.06)     # hit 3 -> confirmed -> on_enter
    assert enters == [next(iter(reg)).track_id]
    assert next(iter(reg)).confirmed is True


def test_default_confirm_is_immediate():
    enters = []

    class WiredPerson(Person):
        def on_enter(self):
            enters.append(self.track_id)

    reg = TrackedRegistry(entity_types=[WiredPerson])  # default confirm_after_hits=1
    reg.consume([_det(1)], ts=0.0)
    assert len(enters) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking_hysteresis.py -q`
Expected: FAIL — `Detection.__init__() got an unexpected keyword argument 'track_id'` (fixed in Task 4) AND registry has no `confirm_after_hits`. (We implement registry now; the `track_id` field lands in Task 4 — temporarily add it here so the test runs; Task 4 finalises the detector path.)

- [ ] **Step 3: Add `track_id` to Detection now (finalised in Task 4) and implement registry**

In `src/projectart/detection/yolo_dots.py`, add to the `Detection` dataclass (after `class_name`):

```python
    track_id: int | None = None
```

In `src/projectart/tracking/entity.py` `__init__`, after `self.class_id = det.class_id`, add:

```python
        self.track_key: int | None = det.track_id
```

Rewrite `src/projectart/tracking/registry.py` to:

```python
"""TrackedRegistry — the smart container.

Two association modes, chosen per frame:
  * If detections carry `track_id` (from a persistent tracker), associate by id
    (exact). This is the robust path — survives big bbox jumps / crossings.
  * Otherwise fall back to greedy IoU per class (legacy default).

Hysteresis:
  * `confirm_after_hits` — on_enter fires only once an entity has been seen this
    many frames (suppresses 1-frame false positives). Default 1 = immediate.
  * `lost_after_s` / `gone_after_s` — coast through brief dropouts before LEAVING
    / GONE. Default None = use the entity class attrs (0.5 / 2.0).
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

from ..detection.yolo_dots import Detection
from .entity import BBox, EntityState, TrackedEntity

log = logging.getLogger(__name__)

DEFAULT_IOU_THRESHOLD = 0.20


class TrackedRegistry:
    def __init__(
        self,
        entity_types: Iterable[type[TrackedEntity]],
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        fallback_type: type[TrackedEntity] | None = None,
        confirm_after_hits: int = 1,
        lost_after_s: float | None = None,
        gone_after_s: float | None = None,
        min_confidence: float = 0.0,
    ):
        self._by_class: dict[str, type[TrackedEntity]] = {}
        for ty in entity_types:
            for name in ty.class_filter():
                if name in self._by_class:
                    log.warning(
                        "duplicate entity_type for class %r: %s shadowed by %s",
                        name, self._by_class[name].__name__, ty.__name__,
                    )
                self._by_class[name] = ty
        self.entities: dict[int, TrackedEntity] = {}
        self.iou_threshold = iou_threshold
        self.fallback_type = fallback_type
        self.confirm_after_hits = max(1, int(confirm_after_hits))
        self.lost_after_s = lost_after_s
        self.gone_after_s = gone_after_s
        self.min_confidence = float(min_confidence)
        self._next_id = 1
        self._by_track: dict[int, int] = {}  # tracker id -> internal track_id

    def consume(self, detections: list[Detection], ts: float | None = None) -> None:
        now = time.monotonic() if ts is None else float(ts)
        dets = [d for d in detections if d.confidence >= self.min_confidence]
        use_track_ids = any(d.track_id is not None for d in dets)
        if use_track_ids:
            self._consume_by_track_id(dets, now)
        else:
            self._consume_by_iou(dets, now)
        self._age_and_confirm(now)
        self._drop_gone()

    # ---- association: tracker ids ----

    def _consume_by_track_id(self, dets: list[Detection], now: float) -> None:
        for det in dets:
            if det.track_id is None:
                continue
            internal = self._by_track.get(det.track_id)
            ent = self.entities.get(internal) if internal is not None else None
            if ent is not None:
                if ent.state in (EntityState.ENTERING, EntityState.LEAVING):
                    ent.state = EntityState.PRESENT
                ent.on_update(det, now)
            else:
                self._spawn(det, now)

    # ---- association: greedy IoU (legacy) ----

    def _consume_by_iou(self, dets: list[Detection], now: float) -> None:
        unmatched = list(dets)
        for entity in list(self.entities.values()):
            best, best_iou = -1, self.iou_threshold
            for i, det in enumerate(unmatched):
                if det.class_id != entity.class_id:
                    continue
                iou = entity.last_bbox.iou(BBox.from_detection(det))
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best >= 0:
                det = unmatched.pop(best)
                if entity.state in (EntityState.ENTERING, EntityState.LEAVING):
                    entity.state = EntityState.PRESENT
                entity.on_update(det, now)
        for det in unmatched:
            self._spawn(det, now)

    # ---- spawn / age / confirm / drop ----

    def _spawn(self, det: Detection, now: float) -> None:
        ent_type = self._resolve_type(det)
        if ent_type is None:
            return
        ent = ent_type(track_id=self._next_id, det=det, ts=now)
        self._next_id += 1
        self.entities[ent.track_id] = ent
        if det.track_id is not None:
            self._by_track[det.track_id] = ent.track_id

    def _age_and_confirm(self, now: float) -> None:
        for ent in self.entities.values():
            lost = self.lost_after_s if self.lost_after_s is not None else ent.LOST_AFTER_S
            gone = self.gone_after_s if self.gone_after_s is not None else ent.GONE_AFTER_S
            missing = now - ent.last_seen_ts
            if not ent.confirmed and ent.hits >= self.confirm_after_hits:
                ent.confirmed = True
                ent.state = EntityState.PRESENT
                ent.on_enter()
            if ent.state == EntityState.PRESENT and missing >= lost:
                ent.state = EntityState.LEAVING
            if missing >= gone:
                ent.state = EntityState.GONE

    def _drop_gone(self) -> None:
        gone = [tid for tid, e in self.entities.items() if e.state == EntityState.GONE]
        for tid in gone:
            ent = self.entities.pop(tid)
            if ent.track_key is not None:
                self._by_track.pop(ent.track_key, None)
            if ent.confirmed:
                ent.on_leave()

    # ---- introspection ----

    def of_class(self, class_name: str) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if e.class_name == class_name]

    def confirmed(self) -> list[TrackedEntity]:
        return [e for e in self.entities.values() if e.confirmed]

    def get(self, track_id: int) -> TrackedEntity | None:
        return self.entities.get(track_id)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities.values())

    def _resolve_type(self, det: Detection) -> type[TrackedEntity] | None:
        ty = self._by_class.get(det.class_name or "")
        if ty is not None:
            return ty
        return self.fallback_type
```

> Note: `_spawn` no longer fires `on_enter` immediately — `_age_and_confirm` fires it once confirmed (default confirm=1 → same frame). Existing `test_tracking.py` expectations (immediate enter, present-after-enter) still hold because confirm defaults to 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking_hysteresis.py tests/test_tracking.py tests/test_scene.py -q`
Expected: PASS (new hysteresis tests + all existing tracking/scene tests green)

- [ ] **Step 5: Commit**

```bash
git add src/projectart/tracking/registry.py src/projectart/tracking/entity.py \
        src/projectart/detection/yolo_dots.py tests/test_tracking_hysteresis.py
git commit -m "feat(tracking): track_id association + confirm/coast hysteresis" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `detection/yolo_dots.py` — persistent tracker path

**Files:**
- Modify: `src/projectart/detection/yolo_dots.py`
- Test: `tests/test_yolo_dots.py` (extend)

**Design:** Add `DotDetector.track(frame)` using `model.track(persist=True, tracker=<yaml>)`. Extract a pure parser `_parse_boxes(boxes, names, with_ids)` so we can test id parsing with a stub (no ultralytics needed).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_yolo_dots.py
from projectart.detection.yolo_dots import Detection, _parse_boxes


class _FakeTensor:
    def __init__(self, arr):
        import numpy as np
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeBoxes:
    def __init__(self, xywh, cls, conf, ids=None):
        self.xywh = _FakeTensor(xywh)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)
        self.id = _FakeTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.xywh.numpy())


def test_parse_boxes_with_track_ids():
    boxes = _FakeBoxes(
        xywh=[[100, 100, 20, 20], [200, 200, 30, 30]],
        cls=[0, 15], conf=[0.9, 0.8], ids=[7, 9],
    )
    dets = _parse_boxes(boxes, {0: "person", 15: "cat"}, with_ids=True)
    assert [d.track_id for d in dets] == [7, 9]
    assert [d.class_name for d in dets] == ["person", "cat"]
    assert dets[0].cx == 100


def test_parse_boxes_without_ids():
    boxes = _FakeBoxes(xywh=[[1, 2, 3, 4]], cls=[0], conf=[0.5], ids=None)
    dets = _parse_boxes(boxes, {0: "person"}, with_ids=True)
    assert dets[0].track_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_yolo_dots.py -q`
Expected: FAIL — `ImportError: cannot import name '_parse_boxes'`

- [ ] **Step 3: Refactor parsing into `_parse_boxes` and add `track()`**

In `src/projectart/detection/yolo_dots.py`, add a module-level helper (after the `Detection` dataclass):

```python
def _parse_boxes(boxes, class_names: dict[int, str], with_ids: bool) -> list["Detection"]:
    """Pure converter from an ultralytics Boxes-like object to Detections.
    Accepts anything exposing .xywh/.cls/.conf/.id with .cpu().numpy()."""
    if boxes is None or len(boxes) == 0:
        return []

    def arr(t):
        return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)

    xywh = arr(boxes.xywh)
    cls = arr(boxes.cls).astype(int)
    conf = arr(boxes.conf)
    ids = None
    if with_ids and getattr(boxes, "id", None) is not None:
        ids = arr(boxes.id).astype(int)
    out: list[Detection] = []
    for i in range(len(xywh)):
        cx, cy, w, h = (float(v) for v in xywh[i])
        cid = int(cls[i])
        out.append(
            Detection(
                class_id=cid, cx=cx, cy=cy, w=w, h=h,
                confidence=float(conf[i]),
                class_name=class_names.get(cid, ""),
                track_id=(int(ids[i]) if ids is not None else None),
            )
        )
    return out
```

Replace the body of `DotDetector.__call__` to use the helper:

```python
    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        self._ensure_loaded()
        results = self._model(frame_bgr, imgsz=self.imgsz, conf=self.conf_thresh, verbose=False)
        if not results:
            return []
        return _parse_boxes(results[0].boxes, self._class_names, with_ids=False)
```

Add a `track()` method and a tracker-yaml map (after `__call__`):

```python
    _TRACKER_YAML = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}

    def track(self, frame_bgr: np.ndarray, tracker: str = "bytetrack") -> list[Detection]:
        """Run detection + persistent tracking. Each Detection carries a stable
        `track_id` across calls (ByteTrack/BoT-SORT)."""
        self._ensure_loaded()
        yaml = self._TRACKER_YAML.get(tracker, "bytetrack.yaml")
        results = self._model.track(
            frame_bgr, persist=True, tracker=yaml,
            imgsz=self.imgsz, conf=self.conf_thresh, verbose=False,
        )
        if not results:
            return []
        return _parse_boxes(results[0].boxes, self._class_names, with_ids=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_yolo_dots.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/detection/yolo_dots.py tests/test_yolo_dots.py
git commit -m "feat(detection): persistent tracker path + pure box parser" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase B — Reactive core

### Task 5: `reactive/config.py` — the config tree

**Files:**
- Create: `src/projectart/reactive/__init__.py`
- Create: `src/projectart/reactive/config.py`
- Create: `src/projectart/reactive/default_config.json`
- Test: `tests/test_reactive_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactive_config.py
from __future__ import annotations

import json

from projectart.reactive.config import ReactiveConfig, load_default


def test_default_config_loads_and_has_box_and_ball():
    cfg = load_default()
    assert "box" in cfg.objects and "ball" in cfg.objects
    assert cfg.objects["box"].kinematic is True
    assert cfg.objects["ball"].kinematic is False
    assert cfg.sim.tick_hz > 0
    assert len(cfg.rules) >= 1


def test_from_dict_round_trip_minimal():
    d = {
        "tracking": {"confirm_after_hits": 5},
        "objects": {"ball": {"radius": 0.1, "shape": "circle", "mass": 2.0, "kinematic": False}},
        "spawn": [{"kind": "ball", "x": 0.5, "y": 0.5}],
        "rules": [{"match": {"class": "person"}, "action": {"spawn": "ball", "behaviors": []}}],
        "physics": {"restitution": 0.5},
        "sim": {"tick_hz": 60},
    }
    cfg = ReactiveConfig.from_dict(d)
    assert cfg.tracking.confirm_after_hits == 5
    assert cfg.objects["ball"].radius == 0.1
    assert cfg.physics.restitution == 0.5
    assert cfg.sim.tick_hz == 60
    assert cfg.rules[0].action["spawn"] == "ball"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reactive_config.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.reactive`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/reactive/__init__.py
"""Reactive objects: data-driven simulation that turns tracked entities into
moving/colliding objects, streamed to the renderer as SceneFrame snapshots."""
```

```python
# src/projectart/reactive/config.py
"""ReactiveConfig — the single JSON-serialisable config tree.

This is the game-engine-portable contract: tracking knobs, world/coordinate
mapping, physics constants, object templates, spawn list, and match->action
rules. No behaviour is hard-coded; everything tunable lives here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..tracking.config import TrackingConfig


@dataclass(slots=True)
class WorldConfig:
    aspect: float = 16 / 9
    cam_to_world: object = "identity"  # "identity" or a 3x3 list (homography)


@dataclass(slots=True)
class PhysicsConfig:
    friction: float = 0.8
    restitution: float = 0.9
    momentum_transfer: float = 1.0
    gravity: tuple[float, float] = (0.0, 0.0)
    default_mass: float = 1.0


@dataclass(slots=True)
class ObjectTemplate:
    kind: str
    radius: float = 0.05
    shape: str = "circle"  # box | circle
    mass: float = 1.0
    color: str = "#39f"
    kinematic: bool = False


@dataclass(slots=True)
class Rule:
    match: dict
    action: dict


@dataclass(slots=True)
class SimConfig:
    tick_hz: int = 30


@dataclass(slots=True)
class ReactiveConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    objects: dict[str, ObjectTemplate] = field(default_factory=dict)
    spawn: list[dict] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    sim: SimConfig = field(default_factory=SimConfig)
    despawn_ms: float = 600.0

    @classmethod
    def from_dict(cls, d: dict) -> "ReactiveConfig":
        w = d.get("world", {})
        p = d.get("physics", {})
        s = d.get("sim", {})
        objects = {
            name: ObjectTemplate(
                kind=name,
                radius=float(o.get("radius", 0.05)),
                shape=o.get("shape", "circle"),
                mass=float(o.get("mass", 1.0)),
                color=o.get("color", "#39f"),
                kinematic=bool(o.get("kinematic", False)),
            )
            for name, o in d.get("objects", {}).items()
        }
        return cls(
            tracking=TrackingConfig.from_dict(d.get("tracking")),
            world=WorldConfig(
                aspect=float(w.get("aspect", 16 / 9)),
                cam_to_world=w.get("cam_to_world", "identity"),
            ),
            physics=PhysicsConfig(
                friction=float(p.get("friction", 0.8)),
                restitution=float(p.get("restitution", 0.9)),
                momentum_transfer=float(p.get("momentum_transfer", 1.0)),
                gravity=tuple(p.get("gravity", (0.0, 0.0))),
                default_mass=float(p.get("default_mass", 1.0)),
            ),
            objects=objects,
            spawn=list(d.get("spawn", [])),
            rules=[Rule(match=r.get("match", {}), action=r.get("action", {})) for r in d.get("rules", [])],
            sim=SimConfig(tick_hz=int(s.get("tick_hz", 30))),
            despawn_ms=float(d.get("despawn_ms", 600.0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReactiveConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_default() -> "ReactiveConfig":
    return ReactiveConfig.from_json(Path(__file__).with_name("default_config.json"))
```

```json
{
  "tracking": {
    "tracker": "bytetrack",
    "conf": 0.25,
    "iou": 0.5,
    "confirm_after_hits": 3,
    "lost_after_s": 0.5,
    "gone_after_s": 2.0,
    "min_confidence": 0.0,
    "velocity_smoothing": { "mincutoff": 1.0, "beta": 0.05 }
  },
  "world": { "aspect": 1.7778, "cam_to_world": "identity" },
  "physics": {
    "friction": 0.8, "restitution": 0.9, "momentum_transfer": 1.0,
    "gravity": [0.0, 0.0], "default_mass": 1.0
  },
  "objects": {
    "box":  { "radius": 0.06, "shape": "box",    "mass": 1.0, "color": "auto", "kinematic": true },
    "ball": { "radius": 0.04, "shape": "circle", "mass": 1.0, "color": "#39f", "kinematic": false }
  },
  "spawn": [ { "kind": "ball", "x": 0.5, "y": 0.5 } ],
  "rules": [
    { "match": { "class": "person" },
      "action": { "spawn": "box", "behaviors": [
        { "follow":   { "gain": 0.5 } },
        { "scale":    { "source": "bbox", "min": 0.03, "max": 0.12 } },
        { "colorize": { "source": "velocity", "mapping": "speed_to_hue" } }
      ] } },
    { "match": { "class": "*" },
      "action": { "spawn": "box", "behaviors": [ { "follow": { "gain": 0.6 } } ] } }
  ],
  "sim": { "tick_hz": 30 }
}
```

> Add `tracking/config.py` (`TrackingConfig`) if not already present — it was specced in Task-adjacent work; create it now with this content:

```python
# src/projectart/tracking/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackingConfig:
    tracker: str = "bytetrack"
    conf: float = 0.25
    iou: float = 0.5
    confirm_after_hits: int = 3
    lost_after_s: float = 0.5
    gone_after_s: float = 2.0
    min_confidence: float = 0.0
    vel_mincutoff: float = 1.0
    vel_beta: float = 0.05

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrackingConfig":
        d = d or {}
        vs = d.get("velocity_smoothing", {})
        return cls(
            tracker=d.get("tracker", "bytetrack"),
            conf=float(d.get("conf", 0.25)),
            iou=float(d.get("iou", 0.5)),
            confirm_after_hits=int(d.get("confirm_after_hits", 3)),
            lost_after_s=float(d.get("lost_after_s", 0.5)),
            gone_after_s=float(d.get("gone_after_s", 2.0)),
            min_confidence=float(d.get("min_confidence", 0.0)),
            vel_mincutoff=float(vs.get("mincutoff", 1.0)),
            vel_beta=float(vs.get("beta", 0.05)),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reactive_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/reactive/__init__.py src/projectart/reactive/config.py \
        src/projectart/reactive/default_config.json src/projectart/tracking/config.py \
        tests/test_reactive_config.py
git commit -m "feat(reactive): config tree (world/physics/objects/rules) + defaults" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `reactive/objects.py` — ReactiveObject

**Files:**
- Create: `src/projectart/reactive/objects.py`
- Test: `tests/test_reactive_objects.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactive_objects.py
from projectart.reactive.config import ObjectTemplate
from projectart.reactive.objects import ReactiveObject


def test_from_template_copies_fields():
    t = ObjectTemplate(kind="ball", radius=0.04, shape="circle", mass=2.0, color="#f00", kinematic=False)
    o = ReactiveObject.from_template(7, t, x=0.5, y=0.5)
    assert (o.id, o.kind, o.radius, o.mass, o.shape, o.color) == (7, "ball", 0.04, 2.0, "circle", "#f00")
    assert (o.x, o.y, o.vx, o.vy) == (0.5, 0.5, 0.0, 0.0)
    assert o.kinematic is False


def test_inv_mass_kinematic_is_zero():
    t = ObjectTemplate(kind="box", mass=1.0, kinematic=True)
    o = ReactiveObject.from_template(1, t, x=0, y=0)
    assert o.inv_mass == 0.0
    t2 = ObjectTemplate(kind="ball", mass=2.0, kinematic=False)
    o2 = ReactiveObject.from_template(2, t2, x=0, y=0)
    assert o2.inv_mass == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reactive_objects.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.reactive.objects`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/reactive/objects.py
"""ReactiveObject — one simulated object (world coords, [0,1])."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import ObjectTemplate


@dataclass(slots=True)
class ReactiveObject:
    id: int
    kind: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.05
    shape: str = "circle"
    mass: float = 1.0
    color: str = "#39f"
    state: str = ""
    alpha: float = 1.0
    angle: float = 0.0
    kinematic: bool = False
    bound_track_id: int | None = None
    age_s: float = 0.0
    behaviors: list = field(default_factory=list)  # list of (name, params)
    despawning: bool = False

    @property
    def inv_mass(self) -> float:
        if self.kinematic or self.mass <= 0:
            return 0.0
        return 1.0 / self.mass

    @classmethod
    def from_template(
        cls, obj_id: int, t: ObjectTemplate, x: float, y: float,
        bound_track_id: int | None = None, behaviors: list | None = None,
    ) -> "ReactiveObject":
        return cls(
            id=obj_id, kind=t.kind, x=x, y=y, radius=t.radius, shape=t.shape,
            mass=t.mass, color=t.color, kinematic=t.kinematic,
            bound_track_id=bound_track_id, behaviors=behaviors or [],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reactive_objects.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/reactive/objects.py tests/test_reactive_objects.py
git commit -m "feat(reactive): ReactiveObject + template instantiation" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `reactive/physics.py` — integration, friction, collision, momentum transfer

**Files:**
- Create: `src/projectart/reactive/physics.py`
- Test: `tests/test_physics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics.py
import math

import pytest

from projectart.reactive.config import PhysicsConfig
from projectart.reactive.objects import ReactiveObject


def _ball(x, y, vx=0.0, vy=0.0, r=0.05, m=1.0):
    return ReactiveObject(id=1, kind="ball", x=x, y=y, vx=vx, vy=vy, radius=r, mass=m)


def test_integrate_moves_and_applies_friction():
    from projectart.reactive.physics import integrate
    p = PhysicsConfig(friction=1.0, gravity=(0.0, 0.0))
    o = _ball(0.0, 0.0, vx=1.0)
    integrate(o, dt=0.1, physics=p)
    assert o.x == pytest.approx(0.1)
    assert o.vx == pytest.approx(0.9)  # 1.0 * (1 - 1.0*0.1)


def test_kinematic_not_integrated():
    from projectart.reactive.physics import integrate
    o = ReactiveObject(id=1, kind="box", x=0.5, y=0.5, vx=5.0, kinematic=True)
    integrate(o, dt=0.1, physics=PhysicsConfig())
    assert (o.x, o.vx) == (0.5, 5.0)


def test_kinematic_imparts_momentum_to_ball():
    from projectart.reactive.physics import resolve_collisions
    p = PhysicsConfig(restitution=1.0, momentum_transfer=1.0)
    hand = ReactiveObject(id=1, kind="box", x=0.50, y=0.5, vx=1.0, radius=0.05, kinematic=True)
    ball = ReactiveObject(id=2, kind="ball", x=0.58, y=0.5, vx=0.0, radius=0.05, mass=1.0)
    resolve_collisions([hand, ball], p)
    assert ball.vx > 0.0          # pushed in +x (swipe direction)
    assert hand.vx == 1.0          # kinematic unaffected


def test_dynamic_pair_conserves_momentum():
    from projectart.reactive.physics import resolve_collisions
    p = PhysicsConfig(restitution=1.0, momentum_transfer=1.0)
    a = ReactiveObject(id=1, kind="ball", x=0.50, y=0.5, vx=1.0, radius=0.05, mass=1.0)
    b = ReactiveObject(id=2, kind="ball", x=0.58, y=0.5, vx=0.0, radius=0.05, mass=1.0)
    p0 = a.mass * a.vx + b.mass * b.vx
    resolve_collisions([a, b], p)
    p1 = a.mass * a.vx + b.mass * b.vx
    assert p1 == pytest.approx(p0, abs=1e-9)


def test_no_collision_when_apart():
    from projectart.reactive.physics import resolve_collisions
    a = _ball(0.1, 0.1, vx=0.0)
    b = _ball(0.9, 0.9, vx=0.0)
    resolve_collisions([a, b], PhysicsConfig())
    assert a.vx == 0.0 and b.vx == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.reactive.physics`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/reactive/physics.py
"""2D physics for reactive objects (world units, [0,1]).

Dynamic objects integrate with velocity + friction + gravity. Circle-circle
collisions resolve with positional separation and an impulse using restitution
and a tunable momentum_transfer scale. Kinematic objects (track-driven) have
inv_mass 0: they impart momentum but are not pushed — the "swipe -> ball flies"
mechanic.
"""
from __future__ import annotations

import math

from .config import PhysicsConfig
from .objects import ReactiveObject


def integrate(obj: ReactiveObject, dt: float, physics: PhysicsConfig) -> None:
    if obj.kinematic:
        return
    gx, gy = physics.gravity
    obj.vx += gx * dt
    obj.vy += gy * dt
    obj.x += obj.vx * dt
    obj.y += obj.vy * dt
    damp = max(0.0, 1.0 - physics.friction * dt)
    obj.vx *= damp
    obj.vy *= damp


def _resolve_pair(a: ReactiveObject, b: ReactiveObject, physics: PhysicsConfig) -> None:
    dx = b.x - a.x
    dy = b.y - a.y
    dist = math.hypot(dx, dy)
    min_dist = a.radius + b.radius
    if dist >= min_dist:
        return
    inv_a, inv_b = a.inv_mass, b.inv_mass
    inv_sum = inv_a + inv_b
    if inv_sum == 0.0:
        return  # both kinematic — nothing to push
    if dist < 1e-9:
        nx, ny, dist = 1.0, 0.0, 1e-9
    else:
        nx, ny = dx / dist, dy / dist
    # Positional separation (split by inverse mass).
    overlap = min_dist - dist
    a.x -= nx * overlap * (inv_a / inv_sum)
    a.y -= ny * overlap * (inv_a / inv_sum)
    b.x += nx * overlap * (inv_b / inv_sum)
    b.y += ny * overlap * (inv_b / inv_sum)
    # Impulse along the normal.
    rvx = b.vx - a.vx
    rvy = b.vy - a.vy
    vel_along = rvx * nx + rvy * ny
    if vel_along > 0:
        return  # separating already
    e = physics.restitution
    j = -(1.0 + e) * vel_along / inv_sum
    j *= physics.momentum_transfer
    a.vx -= j * inv_a * nx
    a.vy -= j * inv_a * ny
    b.vx += j * inv_b * nx
    b.vy += j * inv_b * ny


def resolve_collisions(objects: list[ReactiveObject], physics: PhysicsConfig) -> None:
    n = len(objects)
    for i in range(n):
        for k in range(i + 1, n):
            _resolve_pair(objects[i], objects[k], physics)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_physics.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/projectart/reactive/physics.py tests/test_physics.py
git commit -m "feat(reactive): 2D physics — integrate, friction, collision, momentum transfer" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `reactive/behaviors.py` — follow, scale, colorize

**Files:**
- Create: `src/projectart/reactive/behaviors.py`
- Test: `tests/test_reactive_behaviors.py`

**EntityView** is the small read-only struct the simulator passes to behaviors (built in Task 10). Define it here so behaviors own their input contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactive_behaviors.py
import pytest

from projectart.reactive.behaviors import EntityView, apply_behavior
from projectart.reactive.objects import ReactiveObject


def _view(x=0.5, y=0.5, vx=0.0, vy=0.0, bbox_area=0.04, conf=0.9, dwell=0.0, cls="person"):
    return EntityView(x=x, y=y, vx=vx, vy=vy, bbox_area=bbox_area, confidence=conf,
                      dwell_s=dwell, class_name=cls)


def test_follow_eases_toward_entity():
    o = ReactiveObject(id=1, kind="box", x=0.0, y=0.0, kinematic=True)
    apply_behavior(o, ("follow", {"gain": 0.5}), _view(x=1.0, y=1.0))
    assert o.x == pytest.approx(0.5)
    assert o.y == pytest.approx(0.5)


def test_follow_negative_gain_flees():
    o = ReactiveObject(id=1, kind="box", x=0.5, y=0.5, kinematic=True)
    apply_behavior(o, ("follow", {"gain": -0.5}), _view(x=0.5, y=0.5))
    # at same point, fleeing does nothing; offset entity to verify direction
    o2 = ReactiveObject(id=2, kind="box", x=0.4, y=0.5, kinematic=True)
    apply_behavior(o2, ("follow", {"gain": -0.5}), _view(x=0.5, y=0.5))
    assert o2.x < 0.4  # moves away from entity


def test_scale_from_bbox_clamped():
    o = ReactiveObject(id=1, kind="box", x=0, y=0)
    apply_behavior(o, ("scale", {"source": "bbox", "min": 0.03, "max": 0.12}), _view(bbox_area=1.0))
    assert o.radius == pytest.approx(0.12)
    apply_behavior(o, ("scale", {"source": "bbox", "min": 0.03, "max": 0.12}), _view(bbox_area=0.0))
    assert o.radius == pytest.approx(0.03)


def test_colorize_speed_to_hue_sets_color_and_state():
    o = ReactiveObject(id=1, kind="box", x=0, y=0)
    apply_behavior(o, ("colorize", {"source": "velocity", "mapping": "speed_to_hue"}),
                   _view(vx=2.0, vy=0.0))
    assert o.color.startswith("hsl(")
    o_slow = ReactiveObject(id=2, kind="box", x=0, y=0)
    apply_behavior(o_slow, ("colorize", {"source": "velocity", "mapping": "speed_to_hue"}),
                   _view(vx=0.0, vy=0.0))
    assert o_slow.color != o.color


def test_unknown_behavior_is_noop():
    o = ReactiveObject(id=1, kind="box", x=0.5, y=0.5)
    apply_behavior(o, ("nonsense", {}), _view())
    assert (o.x, o.y) == (0.5, 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reactive_behaviors.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.reactive.behaviors`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/reactive/behaviors.py
"""Parameterized reactive behaviors. Pure functions of (object, params, entity).

A behavior is a tuple ``(name, params)``. The simulator applies the behavior
list of each object every tick. Adding a behavior = adding one branch here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .objects import ReactiveObject


@dataclass(slots=True)
class EntityView:
    """Read-only snapshot of a tracked entity in world space, passed to behaviors."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    bbox_area: float = 0.0          # fraction of frame area [0,1]
    confidence: float = 1.0
    dwell_s: float = 0.0
    class_name: str = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def apply_behavior(obj: ReactiveObject, behavior: tuple[str, dict], entity: EntityView | None) -> None:
    name, params = behavior
    if name == "follow":
        if entity is None:
            return
        gain = float(params.get("gain", 0.5))
        obj.x += (entity.x - obj.x) * gain
        obj.y += (entity.y - obj.y) * gain
    elif name == "scale":
        if entity is None:
            return
        lo = float(params.get("min", 0.03))
        hi = float(params.get("max", 0.12))
        src = params.get("source", "bbox")
        metric = entity.bbox_area if src == "bbox" else entity.confidence
        obj.radius = _clamp(lo + (hi - lo) * _clamp(metric, 0.0, 1.0), lo, hi)
    elif name == "colorize":
        if entity is None:
            return
        mapping = params.get("mapping", "speed_to_hue")
        if mapping == "speed_to_hue":
            speed = math.hypot(entity.vx, entity.vy)
            hue = int(_clamp(speed, 0.0, 1.0) * 270)  # 0 (slow, red) .. 270 (fast, violet)
            obj.color = f"hsl({hue}, 80%, 60%)"
            obj.state = "fast" if speed > 0.3 else "slow"
    # unknown behaviors are silently ignored (forward-compatible config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reactive_behaviors.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/reactive/behaviors.py tests/test_reactive_behaviors.py
git commit -m "feat(reactive): parameterized behaviors — follow, scale, colorize" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `reactive/rules.py` — match → action rule engine

**Files:**
- Create: `src/projectart/reactive/rules.py`
- Test: `tests/test_reactive_rules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactive_rules.py
from projectart.reactive.config import Rule
from projectart.reactive.rules import match_rule


RULES = [
    Rule(match={"class": "person"}, action={"spawn": "box", "behaviors": [{"follow": {"gain": 0.5}}]}),
    Rule(match={"class": "*"}, action={"spawn": "ball", "behaviors": []}),
]


def test_first_matching_rule_wins():
    r = match_rule(RULES, "person")
    assert r.action["spawn"] == "box"


def test_wildcard_fallback():
    r = match_rule(RULES, "cat")
    assert r.action["spawn"] == "ball"


def test_no_match_returns_none():
    r = match_rule([Rule(match={"class": "dog"}, action={})], "cat")
    assert r is None


def test_behaviors_parsed_to_tuples():
    from projectart.reactive.rules import parse_behaviors
    bs = parse_behaviors([{"follow": {"gain": 0.5}}, {"scale": {"min": 0.03}}])
    assert bs == [("follow", {"gain": 0.5}), ("scale", {"min": 0.03})]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reactive_rules.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.reactive.rules`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/reactive/rules.py
"""Rule engine: pick the first rule whose match applies to a tracked class.

A rule's ``match`` supports ``class`` as an exact name or ``"*"`` wildcard.
First match wins (ordered), so specific rules precede the wildcard fallback.
"""
from __future__ import annotations

from .config import Rule


def match_rule(rules: list[Rule], class_name: str) -> Rule | None:
    for rule in rules:
        want = rule.match.get("class", "*")
        if want == "*" or want == class_name:
            return rule
    return None


def parse_behaviors(specs: list[dict]) -> list[tuple[str, dict]]:
    """Convert config behavior dicts (``{"follow": {...}}``) to ``(name, params)``."""
    out: list[tuple[str, dict]] = []
    for spec in specs:
        for name, params in spec.items():
            out.append((name, dict(params)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reactive_rules.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/reactive/rules.py tests/test_reactive_rules.py
git commit -m "feat(reactive): match->action rule engine (first-match-wins)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: `reactive/world.py` — Simulator

**Files:**
- Create: `src/projectart/reactive/world.py`
- Test: `tests/test_reactive_world.py`

**Design:** `Simulator` owns objects + config. `tick(entities, dt)` where `entities` is a list of `EntityView` plus a `track_id` and `class_name`. We pass a tiny `TrackedView` = EntityView + `track_id`. Per tick:
1. For each entity: find its bound object (by `bound_track_id`); if none and a rule matches, spawn (kinematic templates bind to the track). Update kinematic objects: set vel = entity world vel, apply behaviors.
2. Pre-spawned free objects exist from `config.spawn`.
3. Bound objects whose track is gone → start despawn (fade alpha), remove when alpha≈0.
4. Integrate dynamic objects; resolve collisions.
5. Return snapshot (list of ReactiveObject).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactive_world.py
import pytest

from projectart.reactive.behaviors import EntityView
from projectart.reactive.config import ReactiveConfig
from projectart.reactive.world import Simulator, TrackedView


def _cfg():
    return ReactiveConfig.from_dict({
        "objects": {
            "box":  {"radius": 0.06, "shape": "box", "kinematic": True, "color": "auto"},
            "ball": {"radius": 0.04, "shape": "circle", "kinematic": False, "color": "#39f"},
        },
        "spawn": [{"kind": "ball", "x": 0.5, "y": 0.5}],
        "rules": [{"match": {"class": "person"},
                   "action": {"spawn": "box", "behaviors": [{"follow": {"gain": 1.0}}]}}],
        "physics": {"friction": 0.0, "restitution": 1.0, "momentum_transfer": 1.0},
        "sim": {"tick_hz": 30}, "despawn_ms": 100,
    })


def _person(track_id, x, y, vx=0.0, vy=0.0):
    return TrackedView(track_id=track_id,
                       view=EntityView(x=x, y=y, vx=vx, vy=vy, class_name="person"))


def test_prespawn_ball_present():
    sim = Simulator(_cfg())
    snap = sim.tick([], dt=0.033)
    balls = [o for o in snap if o.kind == "ball"]
    assert len(balls) == 1


def test_person_spawns_one_box_that_follows_and_keeps_id():
    sim = Simulator(_cfg())
    sim.tick([_person(1, 0.2, 0.2)], dt=0.033)
    boxes = [o for o in sim.objects if o.kind == "box"]
    assert len(boxes) == 1
    box_id = boxes[0].id
    # follow gain 1.0 -> snaps to entity
    snap = sim.tick([_person(1, 0.8, 0.8)], dt=0.033)
    box = [o for o in snap if o.kind == "box"][0]
    assert box.id == box_id                     # SAME object (morph, not recreate)
    assert (box.x, box.y) == pytest.approx((0.8, 0.8))
    assert box.bound_track_id == 1


def test_departed_track_despawns_box():
    sim = Simulator(_cfg())
    sim.tick([_person(1, 0.5, 0.5)], dt=0.033)
    assert any(o.kind == "box" for o in sim.objects)
    # track gone; advance past despawn_ms
    sim.tick([], dt=0.2)
    sim.tick([], dt=0.2)
    assert not any(o.kind == "box" for o in sim.objects)


def test_fast_box_pushes_ball():
    sim = Simulator(_cfg())
    # Spawn box at the ball, moving fast in +x; ball should gain +x velocity.
    sim.tick([_person(1, 0.46, 0.5, vx=2.0)], dt=0.033)
    sim.tick([_person(1, 0.48, 0.5, vx=2.0)], dt=0.033)
    ball = [o for o in sim.objects if o.kind == "ball"][0]
    assert ball.vx > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reactive_world.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.reactive.world`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/reactive/world.py
"""Simulator — the backend reference simulation.

tick(entities, dt): bind/spawn objects from rules, drive kinematic (track-bound)
objects from entity motion + behaviors, despawn objects whose track left,
integrate dynamic objects, resolve collisions, return the object snapshot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .behaviors import EntityView, apply_behavior
from .config import ReactiveConfig
from .objects import ReactiveObject
from .physics import integrate, resolve_collisions
from .rules import match_rule, parse_behaviors

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackedView:
    track_id: int
    view: EntityView


def _hash_color(name: str) -> str:
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return f"hsl({h % 360}, 75%, 65%)"


class Simulator:
    def __init__(self, config: ReactiveConfig):
        self.config = config
        self.objects: list[ReactiveObject] = []
        self._next_id = 1
        self._bound: dict[int, int] = {}     # track_id -> object id
        self._despawn_age: dict[int, float] = {}  # object id -> seconds despawning
        self._spawn_static()

    def _spawn_static(self) -> None:
        for spec in self.config.spawn:
            t = self.config.objects.get(spec["kind"])
            if t is None:
                log.warning("spawn references unknown kind %r", spec.get("kind"))
                continue
            self.objects.append(ReactiveObject.from_template(
                self._new_id(), t, x=float(spec.get("x", 0.5)), y=float(spec.get("y", 0.5))))

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def tick(self, entities: list[TrackedView], dt: float) -> list[ReactiveObject]:
        seen = {tv.track_id for tv in entities}

        for tv in entities:
            obj = self._object_for_track(tv)
            if obj is None:
                continue
            if obj.kinematic:
                # velocity = entity world velocity (drives momentum transfer)
                obj.vx, obj.vy = tv.view.vx, tv.view.vy
            for beh in obj.behaviors:
                apply_behavior(obj, beh, tv.view)
            if obj.color == "auto":
                obj.color = _hash_color(tv.view.class_name)
            obj.age_s += dt

        # despawn bound objects whose track is gone
        for track_id, obj_id in list(self._bound.items()):
            if track_id in seen:
                continue
            obj = self._by_id(obj_id)
            if obj is None:
                self._bound.pop(track_id, None)
                continue
            obj.despawning = True
            age = self._despawn_age.get(obj_id, 0.0) + dt
            self._despawn_age[obj_id] = age
            frac = age / max(1e-3, self.config.despawn_ms / 1000.0)
            obj.alpha = max(0.0, 1.0 - frac)
            if obj.alpha <= 0.01:
                self.objects.remove(obj)
                self._bound.pop(track_id, None)
                self._despawn_age.pop(obj_id, None)

        for obj in self.objects:
            integrate(obj, dt, self.config.physics)
        resolve_collisions(self.objects, self.config.physics)
        return list(self.objects)

    def _object_for_track(self, tv: TrackedView) -> ReactiveObject | None:
        obj_id = self._bound.get(tv.track_id)
        if obj_id is not None:
            return self._by_id(obj_id)
        rule = match_rule(self.config.rules, tv.view.class_name)
        if rule is None:
            return None
        kind = rule.action.get("spawn")
        t = self.config.objects.get(kind)
        if t is None:
            log.warning("rule spawns unknown kind %r", kind)
            return None
        behaviors = parse_behaviors(rule.action.get("behaviors", []))
        obj = ReactiveObject.from_template(
            self._new_id(), t, x=tv.view.x, y=tv.view.y,
            bound_track_id=tv.track_id, behaviors=behaviors)
        self.objects.append(obj)
        self._bound[tv.track_id] = obj.id
        return obj

    def _by_id(self, obj_id: int) -> ReactiveObject | None:
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reactive_world.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/projectart/reactive/world.py tests/test_reactive_world.py
git commit -m "feat(reactive): Simulator — spawn/bind/behaviors/despawn/physics tick" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase C — Wire + pipeline + mock renderer

### Task 11: `server/protocol.py` — SceneFrame + version bump

**Files:**
- Modify: `src/projectart/server/protocol.py`
- Test: `tests/test_protocol.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_protocol.py
from projectart.server.protocol import PROTOCOL_VERSION, SceneFrame, SceneObject, to_dict


def test_protocol_version_is_2():
    assert PROTOCOL_VERSION == 2


def test_scene_frame_round_trips():
    f = SceneFrame(ts_ms=123, objects=[
        SceneObject(id=1, kind="box", shape="box", x=0.5, y=0.5, vx=0.1, vy=0.0,
                    r=0.06, color="#39f", state="", alpha=1.0, angle=0.0, track_id=7),
    ])
    d = to_dict(f)
    assert d["type"] == "scene_frame"
    assert d["objects"][0]["kind"] == "box"
    assert d["objects"][0]["track_id"] == 7
    assert d["objects"][0]["x"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_protocol.py -q`
Expected: FAIL — `ImportError: cannot import name 'SceneFrame'` and version assert fails

- [ ] **Step 3: Write implementation**

In `src/projectart/server/protocol.py`: change the version constant:

```python
PROTOCOL_VERSION = 2
```

Add the new dataclasses (after `EntityEvent`):

```python
@dataclass
class SceneObject:
    """One reactive object in a SceneFrame. World units: x,y,r in [0,1];
    vx,vy in world-units/sec. `shape` is a render hint (box|circle)."""
    id: int
    kind: str
    shape: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.05
    color: str = "#39f"
    state: str = ""
    alpha: float = 1.0
    angle: float = 0.0
    track_id: int | None = None


@dataclass
class SceneFrame:
    """A full snapshot of all reactive objects for one sim tick. The renderer
    keys visuals on object id and morphs between successive frames."""
    ts_ms: int
    objects: list = field(default_factory=list)
    type: Literal["scene_frame"] = "scene_frame"
```

> `Hello.version` already reports `PROTOCOL_VERSION`, so the renderer sees version 2 automatically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_protocol.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/server/protocol.py tests/test_protocol.py
git commit -m "feat(protocol): SceneFrame/SceneObject + bump PROTOCOL_VERSION to 2" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: `inputs/reactive.py` — ReactiveSource pipeline

**Files:**
- Create: `src/projectart/inputs/reactive.py`
- Test: `tests/test_reactive_source.py`

**Design:** Mirrors `ScenePublisher`. A pure `step(detections, ts) -> SceneFrame` (no camera) drives: registry.consume → build `TrackedView`s for confirmed entities (map cam→world via `CamToWorld`, world velocity by finite-difference of mapped centers kept per track) → `Simulator.tick` → `SceneFrame`. `run()` adds the camera/tracker loop + broadcast.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactive_source.py
from projectart.detection.yolo_dots import Detection
from projectart.reactive.config import ReactiveConfig
from projectart.inputs.reactive import ReactiveSource


def _det(tid, cx, cy):
    return Detection(class_id=0, cx=cx, cy=cy, w=64, h=64, confidence=0.9,
                     class_name="person", track_id=tid)


def _cfg():
    return ReactiveConfig.from_dict({
        "tracking": {"confirm_after_hits": 1},
        "objects": {"box": {"radius": 0.06, "shape": "box", "kinematic": True, "color": "auto"}},
        "spawn": [],
        "rules": [{"match": {"class": "person"},
                   "action": {"spawn": "box", "behaviors": [{"follow": {"gain": 1.0}}]}}],
        "sim": {"tick_hz": 30},
    })


def test_step_emits_scene_frame_with_box_in_world_coords():
    src = ReactiveSource.for_testing(_cfg(), frame_w=640, frame_h=360)
    frame = src.step([_det(1, 320, 180)], ts=0.0)
    assert frame.type == "scene_frame"
    box = [o for o in frame.objects if o.kind == "box"][0]
    # 320/640, 180/360 -> centre of world
    assert abs(box.x - 0.5) < 1e-6 and abs(box.y - 0.5) < 1e-6
    assert box.track_id == 1


def test_step_keeps_same_object_id_across_frames():
    src = ReactiveSource.for_testing(_cfg(), frame_w=640, frame_h=360)
    f1 = src.step([_det(1, 100, 100)], ts=0.0)
    id1 = f1.objects[0].id
    f2 = src.step([_det(1, 500, 300)], ts=0.05)
    assert f2.objects[0].id == id1   # morph, not recreate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reactive_source.py -q`
Expected: FAIL — `ModuleNotFoundError: projectart.inputs.reactive`

- [ ] **Step 3: Write implementation**

```python
# src/projectart/inputs/reactive.py
"""Reactive input source: capture -> tracker -> registry -> simulator -> SceneFrame.

`step()` is pure (no camera) for tests. `run()` adds the live loop. World
coordinates come from CamToWorld; world velocity is a finite-difference of
mapped centres kept per track (correct for identity or homography mapping).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ..capture.yi_rtsp import YiCapture, yi_rtsp_url
from ..detection.yolo_dots import Detection, DotDetector
from ..geometry.mapping import CamToWorld
from ..reactive.behaviors import EntityView
from ..reactive.config import ReactiveConfig, load_default
from ..reactive.world import Simulator, TrackedView
from ..server.protocol import SceneFrame, SceneObject
from ..server.ws import Server
from ..tracking import TrackedRegistry
from ..tracking.builtins import GenericEntity, Person

log = logging.getLogger(__name__)


class ReactiveSource:
    def __init__(
        self,
        canvas_size: tuple[int, int],
        server: Server | None,
        config: ReactiveConfig,
        camera_url_a: str | None = None,
        yolo_weights_path: Optional[str] = None,
        frame_size: tuple[int, int] = (640, 360),
    ):
        self.canvas_size = canvas_size
        self.server = server
        self.config = config
        self.frame_w, self.frame_h = frame_size
        self.camera_url_a = camera_url_a
        self._period = 1.0 / max(1, config.sim.tick_hz)

        self.cam_to_world = CamToWorld.from_config(config.world.cam_to_world)
        self.registry = TrackedRegistry(
            entity_types=[Person],
            fallback_type=GenericEntity,
            confirm_after_hits=config.tracking.confirm_after_hits,
            lost_after_s=config.tracking.lost_after_s,
            gone_after_s=config.tracking.gone_after_s,
            min_confidence=config.tracking.min_confidence,
        )
        self.sim = Simulator(config)
        self._prev_world: dict[int, tuple[float, float, float]] = {}  # track_id -> (x,y,ts)

        self.capture_a: YiCapture | None = None
        self.detector: DotDetector | None = None
        if camera_url_a is not None:
            self.capture_a = YiCapture(url=camera_url_a, name="cam-a")
            self.detector = DotDetector(weights_path=yolo_weights_path)

    @classmethod
    def for_testing(cls, config: ReactiveConfig, frame_w: int, frame_h: int) -> "ReactiveSource":
        return cls(canvas_size=(1920, 1080), server=None, config=config,
                   camera_url_a=None, frame_size=(frame_w, frame_h))

    # ---- pure step (no camera) ----

    def step(self, detections: list[Detection], ts: float) -> SceneFrame:
        self.registry.consume(detections, ts=ts)
        views: list[TrackedView] = []
        for ent in self.registry.confirmed():
            wx, wy = self.cam_to_world(ent.center[0], ent.center[1], self.frame_w, self.frame_h)
            tkey = ent.track_key if ent.track_key is not None else ent.track_id
            vx = vy = 0.0
            prev = self._prev_world.get(tkey)
            if prev is not None:
                pdt = max(1e-3, ts - prev[2])
                vx, vy = (wx - prev[0]) / pdt, (wy - prev[1]) / pdt
            self._prev_world[tkey] = (wx, wy, ts)
            frame_area = float(self.frame_w * self.frame_h)
            bbox_area = (ent.last_bbox.w * ent.last_bbox.h) / max(1.0, frame_area)
            views.append(TrackedView(
                track_id=tkey,
                view=EntityView(x=wx, y=wy, vx=vx, vy=vy, bbox_area=bbox_area,
                                confidence=ent.last_confidence,
                                dwell_s=ts - ent.first_seen_ts, class_name=ent.class_name),
            ))
        objs = self.sim.tick(views, dt=self._period)
        return SceneFrame(
            ts_ms=int(ts * 1000),
            objects=[SceneObject(
                id=o.id, kind=o.kind, shape=o.shape, x=o.x, y=o.y, vx=o.vx, vy=o.vy,
                r=o.radius, color=o.color, state=o.state, alpha=o.alpha, angle=o.angle,
                track_id=o.bound_track_id) for o in objs],
        )

    # ---- live loop ----

    async def run(self) -> None:
        assert self.capture_a is not None and self.detector is not None and self.server is not None
        log.info("reactive source starting (cam-a=%s, tick_hz=%d)",
                 self.capture_a.url, self.config.sim.tick_hz)
        self.capture_a.start()
        loop_t0 = time.monotonic()
        try:
            while True:
                t0 = time.monotonic()
                frame = self.capture_a.latest()
                if frame is None:
                    await asyncio.sleep(self._period)
                    continue
                self.frame_h, self.frame_w = frame.image.shape[:2]
                dets = self.detector.track(frame.image, tracker=self.config.tracking.tracker)
                scene = self.step(dets, ts=time.monotonic() - loop_t0)
                await self.server.broadcast(scene)
                await asyncio.sleep(max(0.0, self._period - (time.monotonic() - t0)))
        finally:
            self.capture_a.stop()


def build_reactive_source(
    canvas_size: tuple[int, int],
    server: Server,
    webcam_a: Optional[str],
    yolo_weights: Optional[str],
    config_path: Optional[str] = None,
) -> ReactiveSource:
    config = ReactiveConfig.from_json(config_path) if config_path else load_default()
    url_a = webcam_a or yi_rtsp_url(host="10.0.0.33", low_res=True)
    return ReactiveSource(canvas_size=canvas_size, server=server, config=config,
                          camera_url_a=url_a, yolo_weights_path=yolo_weights)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reactive_source.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projectart/inputs/reactive.py tests/test_reactive_source.py
git commit -m "feat(inputs): ReactiveSource pipeline (capture->track->sim->SceneFrame)" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: `renderer/reactive.js` + `renderer/reactive.html` — morphing mock

**Files:**
- Create: `renderer/reactive.js`
- Create: `renderer/reactive.html`

**Verification is manual** (browser). No pytest. The key behavior: maintain a visual per object `id`, tween position/size/color toward the latest SceneFrame, remove only when the id is absent for a grace period.

- [ ] **Step 1: Write the renderer page**

```html
<!-- renderer/reactive.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ProjectArt — Reactive (mock)</title>
  <style>
    html, body { margin: 0; height: 100%; background: #0a0a0a; overflow: hidden; cursor: none; }
    canvas { display: block; }
    #status { position: fixed; top: 8px; left: 8px; color: rgba(255,255,255,.55);
      font: 12px ui-monospace, Menlo, monospace; background: rgba(0,0,0,.35);
      padding: 6px 10px; border-radius: 8px; pointer-events: none; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/p5@1.9.4/lib/p5.min.js"></script>
</head>
<body>
  <div id="status">connecting…</div>
  <script src="ws_client.js"></script>
  <script src="reactive.js"></script>
</body>
</html>
```

```javascript
// renderer/reactive.js
// Consumes SceneFrame snapshots and maintains one persistent visual per object id,
// morphing (tweening) position/size/color toward the latest snapshot. Objects are
// created on first sight and removed only after their id is absent for a grace period.
(function () {
  const GRACE_MS = 400;          // keep a visual this long after its id disappears
  const TWEEN = 0.35;            // per-frame easing toward target [0..1]
  const visuals = new Map();     // id -> {x,y,r,color,alpha,shape, tx,ty,tr,tcolor,talpha, lastSeen}

  function onMessage(msg) {
    if (msg.type !== 'scene_frame') return;
    const now = performance.now();
    for (const o of msg.objects) {
      let v = visuals.get(o.id);
      if (!v) {
        v = { x: o.x, y: o.y, r: o.r, color: o.color, alpha: o.alpha, shape: o.shape };
        visuals.set(o.id, v);
      }
      v.tx = o.x; v.ty = o.y; v.tr = o.r; v.tcolor = o.color;
      v.talpha = o.alpha; v.shape = o.shape; v.lastSeen = now;
    }
  }

  function setup() {
    createCanvas(windowWidth, windowHeight);
    rectMode(CENTER);
    window.PA_WS.on('message', onMessage);
  }
  function windowResized() { resizeCanvas(windowWidth, windowHeight); }

  function draw() {
    background(10, 10, 10);
    const now = performance.now();
    const W = width, H = height;
    for (const [id, v] of visuals) {
      if (now - v.lastSeen > GRACE_MS) { visuals.delete(id); continue; }
      v.x += (v.tx - v.x) * TWEEN;
      v.y += (v.ty - v.y) * TWEEN;
      v.r += (v.tr - v.r) * TWEEN;
      v.alpha += ((v.talpha ?? 1) - v.alpha) * TWEEN;
      if (v.tcolor) v.color = v.tcolor;
      const px = v.x * W, py = v.y * H, pr = v.r * Math.min(W, H);
      push();
      noFill();
      const c = color(v.color);
      c.setAlpha(255 * Math.max(0, Math.min(1, v.alpha)));
      stroke(c); strokeWeight(3);
      if (v.shape === 'box') rect(px, py, pr * 2, pr * 2, 6);
      else circle(px, py, pr * 2);
      pop();
    }
  }

  window.setup = setup; window.draw = draw; window.windowResized = windowResized;
})();
```

- [ ] **Step 2: Manual verification (deferred to Task 15)**

The page is verified end-to-end in Task 15 against the live feed. Static check now:
Run: `python -c "import pathlib; assert pathlib.Path('renderer/reactive.js').exists() and pathlib.Path('renderer/reactive.html').exists()"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add renderer/reactive.js renderer/reactive.html
git commit -m "feat(renderer): reactive mock — morph one visual per object id" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: CLI wiring — `--input reactive` + `--reactive-config`

**Files:**
- Modify: `src/projectart/__main__.py`
- Modify: `src/projectart/app.py`

- [ ] **Step 1: Add the CLI flags**

In `src/projectart/__main__.py`, add `reactive` to the `--input` choices list and add a config flag after `--yolo-weights`:

```python
        choices=["mouse", "gloves", "scene", "reactive", "wand", "androidtv"],
```

```python
    p.add_argument("--reactive-config", default=None, help="path to a reactive config JSON")
```

- [ ] **Step 2: Wire the source in `app.py`**

In `src/projectart/app.py`, add a branch in `run()` after the `scene` branch:

```python
        elif self.args.input == "reactive":
            from .inputs.reactive import build_reactive_source

            source = build_reactive_source(
                canvas_size=self.canvas_size,
                server=self._server,
                webcam_a=self.args.webcam_a,
                yolo_weights=self.args.yolo_weights,
                config_path=getattr(self.args, "reactive_config", None),
            )
            await source.run()
```

- [ ] **Step 3: Verify the CLI parses and the full suite is green**

Run: `python -m projectart --input reactive --help >/dev/null && python -m pytest -q`
Expected: help exits 0; all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/projectart/__main__.py src/projectart/app.py
git commit -m "feat(cli): --input reactive + --reactive-config" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: End-to-end demo via the LAN relay + dev helper

**Files:**
- Create: `tools/lan_relay.py` (dev-only helper; documents the Local-Network workaround)
- Modify: `renderer/ws_client.js` (none expected — `?ws=` already supported)

**Design:** On this Mac, third-party binaries can't reach the LAN (see project memory). The relay lets opencv (inside python) reach the camera via Apple's `nc`. This is a dev/deploy helper, not part of the library. Once Local Network permission is granted, point `--webcam-a rtsp://10.0.0.33/ch0_1.h264` directly and skip the relay.

- [ ] **Step 1: Add the relay helper**

```python
# tools/lan_relay.py
"""Dev helper: loopback TCP relay to a LAN host via Apple's /usr/bin/nc.

Works around macOS Local-Network blocking of third-party binaries (python/
opencv/ffmpeg) — see memory/lan-blocked-for-thirdparty-binaries. Run this, then
point the app at rtsp://127.0.0.1:8554/<path>.

    python tools/lan_relay.py --listen 127.0.0.1:8554 --to 10.0.0.33:554
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import threading


def _handle(cli: socket.socket, host: str, port: int) -> None:
    nc = subprocess.Popen(["/usr/bin/nc", host, str(port)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def pump(src_read, dst_write, closer):
        try:
            while True:
                d = src_read(65536)
                if not d:
                    break
                dst_write(d)
        except Exception:
            pass
        finally:
            closer()

    threading.Thread(target=pump, args=(cli.recv, lambda d: (nc.stdin.write(d), nc.stdin.flush()),
                                        lambda: nc.stdin.close()), daemon=True).start()
    threading.Thread(target=pump, args=(nc.stdout.read1, cli.sendall,
                                        lambda: (cli.close(), nc.kill())), daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8554")
    ap.add_argument("--to", default="10.0.0.33:554")
    args = ap.parse_args()
    lh, lp = args.listen.split(":")
    th, tp = args.to.split(":")
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lh, int(lp)))
    srv.listen(8)
    print(f"relay {args.listen} -> {args.to} (nc outbound)", flush=True)
    while True:
        cli, _ = srv.accept()
        _handle(cli, th, int(tp))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the end-to-end demo (manual)**

```bash
# Force RTSP-over-TCP so a single relayed connection carries control+RTP:
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
python tools/lan_relay.py --listen 127.0.0.1:8554 --to 10.0.0.33:554 &
python -m projectart --input reactive \
  --webcam-a rtsp://127.0.0.1:8554/ch0_1.h264 \
  --http-port 8011 --ws-port 8766
# Browser: open http://127.0.0.1:8011/reactive.html?ws=ws://127.0.0.1:8766/
```

Expected: a box appears per confirmed person and **morphs** (glides + resizes) with one stable id as they move; the centre ball gets knocked when a fast-moving box overlaps it. Track ids do not churn.

- [ ] **Step 3: Verify acceptance + full suite**

Run: `python -m pytest -q`
Expected: all tests pass (AC-6).
Confirm visually: AC-1 (one id, no churn), AC-2 (follow/scale/color), AC-3 (ball knocked), AC-7 (box morphs, not recreated).

- [ ] **Step 4: Commit**

```bash
git add tools/lan_relay.py
git commit -m "chore(tools): LAN loopback relay dev helper for camera access" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Stop and report to the user**

The rudimentary reactive-objects system is working end-to-end. Report status and pause for projector integration (the `world.cam_to_world` transform is the seam; identity mock until then).

---

## Self-Review

**Spec coverage:**
- §1 coordinate mapping → Task 1 (`CamToWorld`), used in Task 12. ✔
- §2 stable tracking → Tasks 2 (velocity/hits), 3 (track_id + hysteresis), 4 (tracker path). ✔
- §3 objects/behaviors/physics/rules/world → Tasks 6,8,7,9,10. ✔ (momentum transfer = Task 7; spawn/despawn = Task 10)
- §4 config tree → Task 5 (`ReactiveConfig` + default JSON). ✔
- §5 wire + renderer morph → Tasks 11 (SceneFrame/version) + 13 (reactive.js morph-by-id). ✔
- §6 module layout / pipeline → Task 12 (ReactiveSource) + 14 (CLI). ✔
- §7 testing → each task ships tests; full-suite gate in Tasks 14/15. ✔
- §8 acceptance AC-1..AC-7 → verified in Task 15 (AC-1/2/3/7 visual, AC-4 config-edit, AC-5 Task 1 test, AC-6 suite). ✔
- §9 future (projector/engine) → seam in Task 1 + relay note in Task 15. ✔

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✔

**Type consistency:** `Detection.track_id`, `TrackedEntity.center/velocity/hits/confirmed/track_key`, `TrackedRegistry(confirm_after_hits, lost_after_s, gone_after_s, min_confidence)` + `.confirmed()`, `ReactiveConfig`/`ObjectTemplate`/`Rule`, `ReactiveObject.from_template`/`.inv_mass`, `EntityView`, `apply_behavior((name,params), entity)`, `match_rule`/`parse_behaviors`, `Simulator.tick(list[TrackedView], dt)`, `SceneFrame`/`SceneObject`, `ReactiveSource.step/for_testing/run` — names match across tasks. ✔

**AC-4 note:** "config edit changes behavior" is satisfied by `--reactive-config` (Task 12/14) + the data-driven engine; exercised manually in Task 15.
