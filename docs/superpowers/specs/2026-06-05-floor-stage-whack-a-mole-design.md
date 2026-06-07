# ProjectArt — Floor Stage Platform + Whack-a-Mole — Design

**Status:** Approved (brainstorm) — ready for implementation plan
**Date:** 2026-06-05
**Builds on:** the reactive-objects system (`feat/reactive-objects`) — reuses the
reactive simulator, collision, `SceneFrame`, and the tracking registry.

---

## TL;DR

A reusable **floor-projection game platform**: a canonical normalized **stage**
bracketed by two **manually-adjustable homographies** — one warps the rendered
stage so an **angled projector** lands a clean rectangle on the floor
(stage→projector), one maps the **angled camera** onto the stage (camera→stage) —
plus a manual **height offset** for markers held above the floor. **ArUco markers**
(mallet/foot tags) are tracked through it; **Whack-a-Mole** is the first game built
on top. The RC-car control loop is a separate follow-on spec that reuses this
platform.

This makes concrete the projector-calibration seam deferred from the reactive
spec: `cam_to_world` becomes a real, adjustable **input (camera→stage)** plus a new
**output (stage→projector)** transform.

---

## Goals

- **Correct the geometry of an angled projector** so the stage appears as a clean
  rectangle on the floor — adjustable by dragging the 4 projected corners.
- **Map an angled camera onto the stage** so a tracked marker lands at the right
  floor location — adjustable via the 4 stage corners.
- **Geometry-correct interaction via at-height input calibration:** calibrate
  `camera→stage` with the marker at *playing height* (not flat on the floor), so
  the homography is exact for the plane the marker actually moves in. A residual
  `height_offset` (manual now) absorbs only height *variation*, not the primary
  parallax.
- **Marker tracking** (ArUco) with natively-stable IDs, fast and robust.
- **Whack-a-Mole**: moles projected on the floor; whack = a marker over an active
  mole within its window → score; reuses the reactive sim + collision.
- **Faster camera input** (iPhone via Continuity Camera as a local webcam).
- Everything parameterized (config tree + persisted calibration), reusable by the
  car game.

## Non-Goals (this iteration)

- Automatic projector/camera calibration (manual 4-corner now; auto later).
- Height-aware/stereo parallax correction (single manual offset now).
- The RC-car control loop, BLE, gamepad (separate follow-on spec).
- Multiplayer scoring beyond distinguishing markers by ArUco ID.
- Curved/non-planar floors.

---

## 1. The stage + two adjustable transforms

- **Stage** = normalized `[0,1] × [0,1]`, configurable `aspect`. All game logic,
  moles, marker positions, and collisions live in stage coordinates.
- **`StageCalibration`** (new `geometry/stage.py`) holds:
  - `cam_to_stage`: 3×3 homography mapping camera pixels → stage `[0,1]`.
  - `stage_to_projector`: 3×3 homography mapping stage `[0,1]` → projector/window
    pixels (used by the renderer to pre-warp output).
  - `height_offset`: `(dx, dy)` in stage units, a small residual nudge for height
    *variation* (mallet tilt, different players). NOTE: the primary parallax of a
    marker at height *h* seen by an angled camera is a **radial scaling from the
    camera's nadir** (≈ `h/(c_z−h)` × distance-from-nadir), **not** a constant
    offset — so it is handled by calibrating `cam_to_stage` at playing height (the
    constant-height marker plane is itself related to the camera image by a
    homography), leaving only this small residual.
  - Built from four corner-correspondences each (`cv2.getPerspectiveTransform`);
    identity defaults so the system runs uncalibrated (wrong but visible).
- This makes concrete the reactive spec's placeholder `cam_to_world`: the new
  **floor source** (§5) maps detections cam→stage via `cam_to_stage`
  (+ `height_offset`), and the renderer applies `stage_to_projector`. The existing
  reactive source keeps its simpler `cam_to_world` (migrating it is optional and
  out of scope here).

### Calibration mode
- **Output (stage→projector):** the renderer projects the stage outline + 4
  **draggable corner handles**; the user drags each projected corner until the
  stage looks square/aligned on the floor. The 4 projector-pixel corners define
  `stage_to_projector`.
- **Input (camera→stage), done AFTER the output corners are locked:** with the
  stage outline projected, place the ArUco marker **on a block of the mallet/foot's
  playing height** at each of the 4 projected corners in turn; the captured camera
  positions give the 4 camera-pixel corners → `cam_to_stage`, **exact for the
  playing-height plane**. Calibrating both transforms through the *same* four
  physical projected corners is a nice consistency property — but it means
  **re-dragging the output corners invalidates the input calibration**, so re-do
  input after any output change. A manual nudge sets the residual `height_offset`.
- Persisted to `~/.projectart/calib.json` (extend `CalibrationDoc` with a `stage`
  section: two 3×3 homographies + height offset). Loaded on startup.

## 2. Marker tracking + faster camera

- **`detection/aruco.py`** (cv2.aruco) — detects ArUco markers per frame, returns
  `Marker(id, corners, center)` in camera pixels. Use a **small dictionary
  (`DICT_4X4_50`)** — fewer bits detect faster and more reliably at distance/skew
  than 6×6. ArUco IDs are **natively stable** across frames (no association needed
  — better than ByteTrack for this).
- Markers feed the existing `TrackedRegistry`/reactive pipeline as entities keyed
  on ArUco ID; positions mapped cam→stage (+ height offset) → stage coords +
  velocity (finite-diff, as in the reactive source).
- **`capture/local_cam.py`** — low-latency local capture via
  `cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)` (Continuity Camera / USB webcam),
  same `Frame`/`latest()` interface as `YiCapture` so it's swappable. The Yi RTSP
  source stays as a fallback.
- **Continuity Camera must run with auto-effects OFF** — especially **Center
  Stage**, which reframes per frame and silently breaks the fixed `cam_to_stage`
  homography (it assumes fixed framing). Also avoid Portrait / auto-exposure
  hunting; use a **short shutter with enough light** so a fast mallet swing isn't
  motion-blurred at the whack moment. Documented in setup.

## 3. Whack-a-Mole game logic

- **Moles = reactive objects** (`kind:"mole"`), spawned by a **schedule** at stage
  grid cells; each has an **active window** then auto-despawns (reuses reactive
  spawn/despawn + fade).
- **Whack = the marker's recent trajectory crosses an active mole** within its
  window → `score += points` (once per mole — debounced), mole reacts (squash/flash
  via state), despawns. Test the **swept segment** from the marker's previous to
  current stage position against the mole (not a single-frame point-in-mole), and
  **coast the marker a few frames on detection dropout** using the finite-diff
  velocity we already compute — a fast swing is motion-blurred and may drop exactly
  at contact. Reuses reactive **collision/overlap** + `BehaviorBus`.
- The mole/hit radius is in normalized stage units; with a non-1 `aspect` a
  "circular" tolerance is an **ellipse on the physical floor** — scale by `aspect`
  if true-circular tolerance is wanted.
- **`games/whack_a_mole.py`** holds only game-specific rules: spawn cadence /
  difficulty curve, mole lifetime, grid layout, points, round timer, game-over.
  Data-driven via a `WhackConfig` (spawn_rate, mole_lifetime_s, grid, points,
  round_seconds). The generic platform (stage, markers, reactive sim) stays
  game-agnostic.

## 4. Renderer

- Renders the stage (moles, score/HUD, round timer) to an **offscreen buffer** in
  stage space, then warps that buffer onto the floor via `stage_to_projector`.
  **The warp must be perspective-correct** (a true homography), **not** a
  two-triangle textured quad — affine per-triangle texcoord interpolation bows
  straight lines along the diagonal even when all four corners land perfectly
  (AC-1 would fail corners-right / interior-bowed). Implement as a **fine grid
  tessellation** of the stage quad (cheap, robust) or a fragment-shader homography
  using the homogeneous (`q`) coordinate.
- **Calibration overlay** (toggled by a key): draggable corner handles for both
  homographies + a height-offset nudge; writes back to the backend to persist.
- Consumes `SceneFrame` (moles/markers) plus a small **`GameState`** message
  (score, round timer, phase). New message → **bump `PROTOCOL_VERSION` (2 → 3)**
  with renderer handling.

## 5. Module layout

```
src/projectart/
  geometry/stage.py          # NEW StageCalibration (cam_to_stage, stage_to_projector, height_offset)
  detection/aruco.py         # NEW ArUco marker detector (cv2.aruco)
  capture/local_cam.py       # NEW local/iPhone webcam capture (AVFoundation)
  calibration/persist.py     # extend CalibrationDoc with a `stage` section
  reactive/ (existing)       # reused: objects, physics(collision), world, config
  games/
    __init__.py
    whack_a_mole.py          # NEW mole schedule, hit windows, scoring, round timer
  inputs/floor_game.py       # NEW source: local_cam -> aruco -> registry -> stage -> sim+game -> SceneFrame+GameState
  server/protocol.py         # + GameState; PROTOCOL_VERSION -> 3
renderer/
  floor.html / floor.js      # NEW stage render -> WebGL 4-corner warp; calibration overlay; HUD
tests/
  test_stage.py  test_aruco.py  test_whack_a_mole.py  test_protocol.py (GameState)
```

`--input floor` selects the floor-game source; `--game whack` (default) selects
the game. The LAN relay is not needed here (local camera); it remains for the Yi
cameras.

## 6. Testing (headless where possible)

- `test_stage.py` — `cam_to_stage`/`stage_to_projector` round-trip known corner
  sets; **edge-midpoints and an interior point map through the composed homography
  where predicted** (guards against affine/perspective mistakes creeping into the
  texcoord math); `height_offset` applies correctly; identity defaults.
- `test_aruco.py` — generate a synthetic ArUco marker image (cv2.aruco), detect it,
  assert id + center (no camera needed).
- `test_whack_a_mole.py` — spawn schedule cadence; a marker over an active mole in
  its window scores once (not twice); a marker over an expired/empty cell scores
  nothing; round timer / game-over.
- `test_protocol.py` — `GameState` round-trips; `PROTOCOL_VERSION == 3`.
- Renderer warp + calibration UI are verified manually with the projector.

## 7. Acceptance criteria (rudimentary "done")

- **AC-1** Dragging the 4 projected corners makes the stage land as a clean
  rectangle on the floor from the angled projector, **with straight stage lines
  staying straight across the interior** (perspective-correct warp), persisted.
- **AC-2** With `camera→stage` calibrated **at playing height**, a marker moved
  anywhere on the stage reports the correct stage position within tolerance
  **across the whole area** (not just near one tuned point); the residual
  height-offset nudge cleans up what's left.
- **AC-3** ArUco marker is tracked with a **stable id** across frames, mapped to
  stage coords.
- **AC-4** Moles spawn/expire on schedule; a whack (marker over an active mole in
  its window) scores exactly once and squashes the mole; misses don't score.
- **AC-5** Score/timer render via `GameState`; calibration persists across runs.
- **AC-6** Editing `WhackConfig` (spawn rate, lifetime, grid, points) changes the
  game with no code edit.
- **AC-7** All new logic tests pass headless; full suite green.
- **AC-8** End-to-end loop latency (capture→detect→sim→render→project) is measured
  and reported, within target (≈ ≤ 100 ms); a real mallet swing registers as a hit.
  Continuity Camera Center Stage / auto-effects are off (fixed framing verified).

## 8. Future / follow-on

- **RC-car control loop spec** layers on this platform: ArUco tag on the car
  (reuses §2), car-over-powerup via reactive collision (reuses §3 pattern), plus
  the new BLE/gamepad control loop (NUS RX `6e400002` on `Tenka6B34`).
- **Auto-calibration** (detect a projected ChArUco) replaces manual corners.
- **Auto height/parallax** (stereo or known marker size → height) replaces the
  manual offset.

## 9. Resolved decisions

- Geometry correction = two manually-adjustable 4-corner homographies (output
  stage→projector + input camera→stage), **input calibrated at playing height**
  (homography exact for that plane) + small residual `height_offset`. ✔
- Output warp is **perspective-correct** (grid tessellation or shader), not
  two-triangle affine. ✔
- Camera angled/off to the side; markers tracked via **ArUco** (`DICT_4X4_50`,
  stable IDs). ✔
- Whack detection = marker's **swept trajectory** vs active mole + velocity-coast on
  dropout + per-mole debounce. ✔
- Faster camera = **iPhone Continuity Camera** as a local webcam, **auto-effects
  (Center Stage) OFF**; end-to-end latency measured. ✔
- Build order = **floor platform + whack-a-mole first**; car-control loop follow-on. ✔
- Calibration UX = **drag projected corners** (output, locked first) + **marker at
  playing height on each corner** (input, redone after any output change). ✔
