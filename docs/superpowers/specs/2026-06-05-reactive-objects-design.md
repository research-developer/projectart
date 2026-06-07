# ProjectArt — Reactive Objects & Stable Tracking — Design

**Status:** Approved (brainstorm) — ready for implementation plan
**Date:** 2026-06-05
**Spec for:** a parameterized, backend-authoritative reactive-object system driven by
stable per-object tracking, with a throwaway browser mock until the projector /
game engine is integrated.

---

## TL;DR

The camera sees objects; YOLO + a persistent tracker assign each a stable id from
entrance to departure; a backend **reference simulator** turns tracked entities into
**reactive objects** that move/change/collide according to a single **config tree**;
the backend streams object snapshots over WebSocket; a small p5 mock renders them.

Everything tunable lives in one JSON-serializable config so the same parameters
later drive a game engine and a projector (the projector is just a swapped
coordinate transform).

---

## Goals

- A single real-world object keeps **one track id** from frame entrance to departure
  (no churn / no "brand new object" mid-presence). This is the primary fix — the
  live demo churned ids to 24+ on a static scene.
- **Reactive objects** that react to what's seen via parameterized primitives:
  follow/anchor, scale, color/state, spawn-on-enter/despawn-on-leave.
- **Momentum & collision:** track-driven objects carry velocity and transfer
  momentum to free virtual objects on contact (swipe a hand → ball flies off).
- **Parameterize as much as possible** — one config tree is the source of truth, so
  behavior changes need no code edit and the config ports to a game engine.
- Backend is the authoritative **reference simulator**; the renderer (and later the
  engine) is a thin consumer of object snapshots.
- Rudimentary-but-working is the bar for this iteration; projector mapping is mocked.

## Non-Goals (this iteration)

- 3D physics; rotational dynamics beyond a passive `angle`.
- Real projector calibration (mock identity transform now; real homography later).
- Replacing the renderer with the actual game engine (mock p5 visualizer only).
- Multi-camera / stereo for this feature (single cam feed; relay transport).
- Networked/multi-client authority or rollback.

---

## 1. Coordinate model & mapping

- Detections are camera pixels (e.g. 640×360). Reactive objects live in a
  **normalized world space `[0,1] × [0,1]`** (resolution-independent). A configurable
  `world.aspect` records the intended physical aspect for non-distorting render.
- `geometry/mapping.py` provides a pluggable `CamToWorld` transform:
  - **Now:** linear/identity — cam pixel `(px,py)` → `(px/W, py/H)`.
  - **Later (projector):** a 3×3 homography loaded from config; "integrating the
    projector" = setting that matrix, no code change.
- All wire-protocol positions/velocities/radii are in **world units**
  (`vx,vy` = world-units per second). One number space for renderer, engine, projector.

## 2. Stable tracking (kill the flicker)

- `detection/yolo_dots.py` gains a tracking path using ultralytics' built-in tracker
  (`model.track(persist=True)`, ByteTrack by default; BoT-SORT selectable). Stays in
  the only module permitted to import ultralytics.
  - `Detection` gains an optional `track_id: int | None`.
- `TrackedRegistry` keys lifecycle on `track_id` when present (falls back to the
  existing greedy-IoU association when the detector yields no id), plus **hysteresis**:
  - `confirm_after_hits` — a track is not surfaced (no `enter`) until seen this many
    consecutive frames (suppresses 1-frame false positives).
  - `lost_after_s` / `gone_after_s` — coast through brief dropouts before `leave`.
  - optional `min_confidence` gate.
- `TrackedEntity` gains **world-space velocity** `(vx, vy)`, smoothed with the existing
  1€ filter, computed from successive world positions. This feeds momentum transfer.

## 3. Reactive objects, physics & rules

New isolated module `reactive/`, one purpose per file:

- `objects.py` — `ReactiveObject` and `ObjectTemplate`.
  - `ReactiveObject`: `id, kind, pos(x,y), vel(vx,vy), radius, shape, mass, color,
    state, alpha, angle, kinematic: bool, bound_track_id: int | None, age_s, behaviors`.
    `shape` is a render hint (`box` | `circle`); `radius` is the collision half-extent
    and base render size.
  - `kinematic=True` → bound to a track: each tick its position eases toward the
    entity's world position (by `follow.gain`) and its **velocity is set to the
    entity's smoothed world velocity** (so collision impulses reflect real hand/body
    speed); treated as infinite mass in collisions. `kinematic=False` → free dynamic
    body integrated by `physics.py`.
- `behaviors.py` — pure, parameterized primitives applied per tick to an object given
  its bound entity (if any):
  - `follow(gain, offset)` — ease object pos toward `entity.world_pos`.
  - `scale(source: bbox|confidence, min, max)` — set radius from entity metric.
  - `colorize(source: class|velocity|dwell, mapping)` — set color/state.
  - lifecycle (`spawn`/`despawn`) handled by the rule engine, not per-tick behaviors.
- `physics.py` — pure 2D integrator + collisions:
  - Integrate: `pos += vel*dt`; `vel *= (1 - friction*dt)` (clamped ≥ 0).
  - Optional `gravity` (default 0).
  - Circle–circle collision: on overlap, positional separation + impulse-based
    velocity resolution using restitution `e` and `momentum_transfer` coefficient.
    Impulse `j = -(1+e)·(relVel·n) / (1/m1 + 1/m2)`; for a kinematic body `1/m → 0`,
    so a fast-moving track-driven object imparts velocity to a free object while
    itself unaffected. `momentum_transfer` scales the impulse for tunable "feel".
- `rules.py` — data-driven rule engine. Each rule: `match {class glob, optional zone,
  optional condition} → action {spawn template | bind existing, attach behaviors with
  params}`. **Rules are evaluated in order; the first matching rule per track wins
  (one spawn per track)** — so specific rules precede the `"*"` fallback. Spawn rules
  fire on track `enter`; behaviors apply every tick while present; despawn (fade
  `alpha→0` over `despawn_ms`) on `leave`. Config may also pre-spawn static free
  objects (e.g. a ball) at startup via `spawn`.
- `world.py` — `Simulator`: holds objects + config; `tick(entities, dt) → list[ReactiveObject]`
  snapshot. Deterministic; no I/O. This is the reference implementation a game engine
  can later replace or re-run.

## 4. Config tree (the parameterization)

`reactive/config.py` — dataclasses, JSON-loadable. This is the portable artifact.

```jsonc
{
  "tracking": {
    "tracker": "bytetrack",            // bytetrack | botsort
    "conf": 0.25, "iou": 0.5,
    "confirm_after_hits": 3,
    "lost_after_s": 0.5, "gone_after_s": 2.0,
    "min_confidence": 0.0,
    "velocity_smoothing": { "mincutoff": 1.0, "beta": 0.05 }
  },
  "world": { "aspect": 1.7778, "cam_to_world": "identity" },  // or 3x3 homography
  "physics": {
    "friction": 0.8, "restitution": 0.9, "momentum_transfer": 1.0,
    "gravity": [0.0, 0.0], "default_mass": 1.0
  },
  "objects": {
    "box":  { "radius": 0.06, "shape": "box",    "mass": 1.0, "color": "auto", "kinematic": true },
    "ball": { "radius": 0.04, "shape": "circle", "mass": 1.0, "color": "#39f", "kinematic": false }
  },
  "spawn": [ { "kind": "ball", "x": 0.5, "y": 0.5 } ],   // static free objects
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

A default config ships in-repo; `--reactive-config PATH` overrides it.

## 5. Wire protocol & renderer mock

- New snapshot message (one per sim tick):
  ```
  SceneFrame {
    type: "scene_frame", ts_ms: int,
    objects: [ { id, kind, shape, x, y, vx, vy, r, color, state, alpha, angle,
                 track_id? } ]   // world units; x,y,r in [0,1]
  }
  ```
- This changes the wire format → **bump `PROTOCOL_VERSION` 1 → 2** with matching
  renderer handling (per CLAUDE.md). `EntityEvent` remains available for the existing
  scene overlay; the reactive renderer consumes `SceneFrame`.
- `renderer/reactive.js` — consumes `SceneFrame` and maintains a **persistent visual
  per object `id`**: it creates a shape on first sight, **morphs** it (tweens position,
  size, color, alpha between snapshots) while the id persists, and removes it only on
  true departure (id absent past a grace period). Because tracking is stable, a tracked
  object's box **glides and resizes across the screen as one element instead of being
  recreated each frame**. Render shape is per `shape` hint (`box` → rectangle, `ball`
  → circle). Served via a small reactive demo page; throwaway mock until projector/engine.

## 6. Pipeline & module layout

```
src/projectart/
  detection/yolo_dots.py      # + track(persist) path; Detection.track_id
  geometry/mapping.py         # NEW  cam_px -> world transform (identity | homography)
  tracking/
    entity.py                 # + world-space velocity (1€-smoothed)
    registry.py               # key on track_id + confirm/coast hysteresis (params)
    config.py                 # NEW  TrackingConfig
  reactive/                   # NEW module
    config.py                 # ReactiveConfig (world/physics/objects/spawn/rules/sim)
    objects.py                # ReactiveObject, ObjectTemplate
    behaviors.py              # follow, scale, colorize (pure, parameterized)
    physics.py                # integrate, friction, circle collision + momentum xfer
    rules.py                  # match -> action rule engine
    world.py                  # Simulator.tick(entities, dt) -> snapshot
  server/protocol.py          # + SceneFrame; PROTOCOL_VERSION -> 2
  inputs/reactive.py          # NEW  ReactiveSource: capture->track->registry->sim->broadcast
renderer/
  reactive.js                 # NEW  consume SceneFrame, draw + tween
  reactive.html               # NEW  mock demo page (or wire into index.html)
tests/
  test_mapping.py  test_tracking_velocity.py  test_physics.py
  test_reactive_rules.py  test_reactive_behaviors.py  test_protocol.py (SceneFrame)
```

`--input reactive` selects the new source. The loopback `nc` relay remains the
transport workaround until Local Network permission is granted (deployment detail,
not part of the repo's logic).

## 7. Testing (all headless, per repo convention)

- `test_physics.py` — integration, friction decay, circle collision separation, and
  **momentum transfer** (kinematic→dynamic gives expected post-impulse velocity;
  dynamic↔dynamic conserves momentum within tolerance).
- `test_tracking_velocity.py` — synthetic track positions → correct smoothed velocity.
- `test_reactive_rules.py` / `test_reactive_behaviors.py` — rule matching + each
  behavior given synthetic entities; spawn/despawn lifecycle.
- registry hysteresis — `confirm_after_hits` suppresses 1-frame tracks; coast across
  dropouts keeps one id (extends `test_tracking.py`).
- `test_mapping.py` — identity and a known homography map sample points correctly.
- `test_protocol.py` — `SceneFrame` round-trips; `PROTOCOL_VERSION == 2`.

## 8. Acceptance criteria (rudimentary "done")

- **AC-1** A synthetic/live track keeps one id entrance→departure across ≥10 s with no
  churn (`confirm`/`coast` verified in tests + observed on the relay feed).
- **AC-2** On the live feed, a reactive object follows + scales + colorizes a tracked
  entity per config, visible in the mock renderer.
- **AC-3** A free `ball`, overlapped by a fast track-driven object, gains velocity in
  the swipe direction and coasts with friction — visible + unit-tested.
- **AC-4** Editing the JSON config (follow→flee gain sign, ball mass/restitution, a
  spawn rule) changes behavior with no code edit.
- **AC-5** `cam_to_world` is config-swappable (identity now; a mock homography remaps
  coordinates) — unit tested.
- **AC-6** All new logic tests pass headless; full suite green.
- **AC-7** A tracked object renders as a **single persistent box keyed by id that morphs
  (position + size + color) across the screen for its whole lifetime** — no per-frame
  pop-in/recreate; the visual is created/destroyed only on true enter/leave.

## 9. Future integration notes

- **Projector:** set `world.cam_to_world` to the calibrated homography (from the M5
  calibration). The mock identity is the only thing that changes. "Map what you think
  you're saying" = render `SceneFrame` through the projector with the real transform.
- **Game engine:** consume `SceneFrame` directly, or re-run the same `ReactiveConfig`
  in-engine and treat the backend sim as the reference. The config tree is the contract.

## 10. Resolved decisions

- World space = normalized `[0,1]` (engine portability). ✔
- Backend runs the reference physics now; engine may re-run later from the same config. ✔
- Tracking = ultralytics persistent tracker + registry hysteresis. ✔
- Reactions = backend, data-driven. ✔
- Behaviors = follow, scale, color/state, spawn/despawn, + momentum/collision transfer. ✔
