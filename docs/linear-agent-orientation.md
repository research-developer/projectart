# ProjectArt — Linear Agent Orientation

Drop this whole document into a Linear Agent prompt (alongside `PRD.md`) so the agent has full context before it touches issues. Updated 2026-05-09.

---

## 1. What you're working on

**ProjectArt** is an interactive room/wall system. A projector (or screen) shows a canvas; two ceiling-mounted Yi cameras see what's in front of it; YOLO detects objects of interest (people, cats, eventually hands and dot-glove fingertips); a tracked-object registry instantiates / updates / disposes per-object state and fires behaviors (sounds, screen overlays, drawing strokes).

Three loosely coupled layers, communicating over a localhost WebSocket:

```
[Yi cams 10.0.0.33 + .34] ──► [Python backend] ──ws──► [browser renderer]
                                  │
                                  ├── capture (RTSP, low-latency)
                                  ├── detection (YOLO + stereo)
                                  ├── geometry (wall plane, 1€ filter, homography)
                                  ├── tracking (TrackedEntity + Registry + BehaviorBus)
                                  ├── inputs   (mouse | gloves | wand)
                                  └── server   (WS + static HTTP)
```

Originally framed as a kid-friendly drawing wall (dot gloves + projector); the OOP tracking layer is the generalisation that lets the same plumbing recognise cats, people, and arbitrary objects with rule-based behaviours.

The full design is in `PRD.md` (synthesises three earlier research PRDs: WallSketch, ProjectorCanvas, drawmagic).

---

## 2. Linear team & project

- **Team**: `Imajn` (key `IMA`)
- **Project**: *ProjectArt — interactive wall drawing* (id `72866694-1a3c-4e44-9b5d-1d3a31da0ca9`)
- **Issues seeded**: IMA-203 … IMA-213 (M0 scaffold through optional N-tier features). See the project page for current state.
- **Conventional issue title pattern**: `Mn — short imperative title` for milestones (`M0 — repo scaffold + boots…`); plain imperative for tactical work (`Implement capture/yi_rtsp.py …`).

When creating new issues for emerging work, prefer:
- `feat(<area>): <thing>` to describe what
- DoD as a bulleted "what done looks like" at the end of the description
- Link the spec section if relevant: `docs/superpowers/specs/2026-05-09-projectart-design.md §3.1`

---

## 3. Repo at a glance

```
projectart/
├── PRD.md                                         (synthesis design doc)
├── docs/
│   ├── linear-agent-orientation.md                (this file)
│   └── superpowers/specs/2026-05-09-projectart-design.md
├── src/projectart/
│   ├── __main__.py    app.py    config.py
│   ├── capture/       yi_rtsp.py             low-latency RTSP, Q=1 newest-only
│   ├── detection/     yolo_dots.py           ultralytics wrapper, --yolo-weights
│   │                  stereo.py              StereoRig + correspondence + triangulate
│   ├── geometry/      wall_plane.py          plane fit + UV basis + contact
│   │                  filter_one_euro.py     1€ adaptive low-pass
│   ├── tracking/      entity.py              TrackedEntity + BBox + EntityState
│   │                  registry.py            TrackedRegistry (smart container)
│   │                  builtins.py            Cat, Person, GenericEntity
│   │                  events.py              BehaviorBus (pub/sub)
│   ├── calibration/   persist.py             pydantic CalibrationDoc + atomic save
│   ├── inputs/        mouse.py | gloves.py
│   ├── server/        protocol.py            wire format dataclasses
│   │                  ws.py                  asyncio WS + static HTTP
│   └── integrations/  androidtv.py           (stub — only if projector source is AndroidTV)
├── renderer/          index.html, sketch.js, ws_client.js
├── tests/             pytest, ~44 passing
├── training/          (planned) datasets, retraining scripts
└── research-notes/    original 3 PRDs
```

`pyproject.toml` is the source of truth for deps. `pytest pythonpath=src` so tests import without an install. ruff is configured.

---

## 4. Where things stand right now

### Working end-to-end
- `python -m projectart --input mouse` opens the browser canvas; mouse drag draws lines via the WS contract.
- Wire format (`Hello`, `PointerEvent`, `HudAnchorEvent`, `CommandEvent`) is the contract every input mode plugs into.
- Tracking layer: detections in → entities out, lifecycle transitions verified, BehaviorBus dispatches without coupling.

### Built but not yet wired into the live pipeline
- `capture/yi_rtsp.py` — RTSP thread; URL helper for the standard yi-hack-v5 pattern.
- `detection/yolo_dots.py` — Ultralytics wrapper, falls back to `yolov8n.pt` when `--yolo-weights` is unset.
- `detection/stereo.py` — class-id correspondence + cv2 triangulation.
- `geometry/wall_plane.py` — plane fit + UV basis + signed distance + contact gating.
- `geometry/filter_one_euro.py` — 1€ adaptive low-pass.
- `calibration/persist.py` — pydantic `CalibrationDoc` schema + atomic save/load.
- `inputs/gloves.py` — single-cam pipeline (placeholder identity homography until M5 wizard lands).
- `tracking/*` — full registry + entity + behaviors + bus.

### Not yet built
- M3 wiring: stereo into `inputs/gloves.py` (modules exist, not stitched).
- M4 floating HUD ring in `renderer/hud.js`.
- M5 calibration wizards (`wizard_4corner.py`, `wizard_charuco.py`).
- M6 polish: p5.brush.js brushes, sounds, save-PNG, undo, clear.
- N1 ArUco wand fallback.
- N3 AndroidTV ADB integration.
- Training pipeline retraining scaffold (planned in `training/`).

---

## 5. The "6× memory / 40% faster" YOLO question

The brief mentioned a custom-trained YOLO with these gains. Tracked down via the user's mounted drive at `/Volumes/research-developer/CatMorph/.worktrees/yolo/`:

- **The gain is in TRAINING cost, not inference speed.** CatMorph monkey-patches YOLO's loss path to use all-integer Z[√3] arithmetic. f32 features 16.5 MB → 4-bit packed 2.8 MB (6×). Standard YOLO26s 53.2 s/epoch → full rational 31.6 s/epoch (-41%).
- **Trained model is ordinary YOLO weights.** No special inference path required.
- **Dataset**: COCO128 (so person & cat are first-class).
- **Production action**: M2 is wired against `yolov8n.pt` as the safe stand-in (loaded by Ultralytics on first use). When the user supplies confirmed weights, swap via `--yolo-weights /path/to.pt`. See Linear IMA-210.

---

## 6. The "tracked-object container" pipeline

User's vision (paraphrased): "YOLO detects an object, sends the event to the object that's being tracked, or to the container which can either instantiate a new object to be tracked or cue some other trigger. If it sees a cat, make a noise and show something on screen, then track until gone, then make another sound and clear the overlay."

Implemented in `src/projectart/tracking/`:

```python
from projectart.tracking import TrackedRegistry
from projectart.tracking.builtins import Cat, Person
from projectart.tracking.events import BehaviorBus

bus = BehaviorBus()
bus.on("cat.appeared", lambda *, entity: play_meow_overlay(entity))
bus.on("cat.left",    lambda *, entity: dismiss_meow_overlay(entity))

registry = TrackedRegistry(entity_types=[Cat, Person])

# Each frame, after YOLO inference:
registry.consume(detections, ts=now)
```

`TrackedEntity` subclasses can either subclass and override hooks, or use `attach_callbacks(Cat, on_enter=...)` for ad-hoc behaviour without a class.

Lifecycle: `ENTERING → PRESENT → LEAVING → GONE` with re-acquire (a `LEAVING` entity that reappears within `GONE_AFTER_S` returns to `PRESENT`).

For dense scenes the simple greedy-IoU association can be swapped for a real tracker (ByteTrack/SORT) without changing the public interface — that's the unit test surface.

---

## 7. Conventions

- **Python 3.11**, `pyproject.toml`, ruff (E,F,I,B,UP). 100-col max. Type-hint public funcs.
- **Hot paths use numpy ops, not Python loops.** Filter, raycast, stereo, association.
- **Lazy imports for cv2 / ultralytics** so modules load in test envs without those deps.
- **Logging via `logging.getLogger(__name__)`** — no `print()` in library code.
- **Tests in `tests/`, pytest.** Geometric / protocol / tracking modules MUST be testable without a display, camera, or YOLO weights.
- **Conventional commits.** `feat(area): …`, `refactor(area): …`, `docs: …`, `test: …`. Co-author trailer is fine, not required.
- **Calibration** lives at `~/.projectart/calib.json`. Schema in `calibration/persist.py`.
- **Cameras**: `10.0.0.33` (cam A), `10.0.0.34` (cam B). yi-hack-v5. Prefer the low-res RTSP stream for inference.

---

## 8. Open questions

1. **Projector**: confirmed mounted? Throw distance? Aspect ratio? Affects calibration wizards (M5).
2. **Gloves**: do dot-marker gloves exist or do we spec/build them? Recommended pattern in PRD §14: dark glove, 5 saturated finger-tip dots (magenta/cyan/lime/yellow/orange), one white back-of-hand dot.
3. **Android TV**: is the projector source an Android TV stick? If yes → wire `integrations/androidtv.py` (N3). If no → drop §9 entirely.
4. **YOLO weights confirmation**: see §5 above + IMA-210.

---

## 9. What "good" looks like for new issues

When the human + Linear Agent collaborate to create new issues, useful patterns:

- **One vertical slice per issue.** Anything bigger is a milestone — break it down.
- **DoD has at least one "the user can …" sentence**, not just "the function exists".
- **Reference the PRD section** where the requirement comes from — keeps drift visible.
- **Mark issues that need the live cameras/projector** with a label like `needs-hardware` so they're scheduled for in-person sessions, not async work.
- **Track the "this is built but not wired" gap explicitly** — it's currently the biggest source of "we have it, why doesn't it work?" surprise.

---

## 10. Useful entry points

- Run the system: `pip install -e . && python -m projectart --input mouse`
- Run tests: `python -m pytest tests/ -q`
- Read first: `PRD.md` → this doc → `tests/test_tracking.py` (shows the OOP layer in action) → `src/projectart/server/protocol.py` (wire format)

---

## 11. Things this agent should NOT do

- Don't merge feature branches into `main` automatically — humans gate that.
- Don't change the wire protocol (`server/protocol.py`) without bumping `PROTOCOL_VERSION` and updating renderer-side handling.
- Don't push uncommitted, untested code. If tests fail, fix them or write a `WAKEUP_NOTE.md`.
- Don't replace the simple greedy-IoU registry association with ByteTrack/SORT without first confirming there's an actual scene-density problem in the field.
- Don't import `cv2`, `ultralytics`, or `pynput` outside their dedicated modules.
