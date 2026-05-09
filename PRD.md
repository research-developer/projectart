# ProjectArt — Combined Design / PRD

**Status:** Draft (synthesis of three overnight research PRDs)
**Date:** 2026-05-09
**Codename:** ProjectArt (working title — cf. WallSketch / ProjectorCanvas / drawmagic in source notes)

---

## TL;DR

A kid-friendly interactive wall (or floor, if no projector) drawing system. The player wears **dot-marker gloves**; two ceiling-mounted Yi cameras (10.0.0.33, 10.0.0.34) see the dots from different angles and a custom YOLO model detects them. Dot 3D positions are triangulated, intersected with a calibrated wall plane, and rendered as ink on a projected canvas. A small **transparent HUD follows the active hand** for tool/color/brush selection — no off-canvas menus. Browser renderer (p5.js) for organic brushes; Python backend for capture, inference, geometry. Pluggable input layer means we can ship the same UX without a projector (drawing on a screen) and without dot gloves (single-cam ArUco wand fallback).

This document combines and trims:
- `research-notes/deep-research-report.md` (WallSketch — pragmatic IR-pen-first PRD)
- `research-notes/gemini-instructions.md` (ProjectorCanvas — touchless YOLO+MediaPipe PRD)
- `research-notes/gemini-research-report.md` (drawmagic — Quest-controller-first PRD)

The combined PRD takes the best of each:
- **Stack & architecture** (decoupled Python ↔ browser, WebSocket, p5.brush.js) → from ProjectorCanvas
- **Calibration math, 1€ filter, pluggable input, brush taxonomy, milestone discipline** → from drawmagic
- **Hardware-realism and milestone phasing** → from WallSketch

The two **new ideas not in any source PRD** that drive this design:
1. **Stereo from two cameras** — solves the contact-detection problem the prior PRDs all admit they don't have a clean answer to.
2. **Dot gloves** — bypass the bare-hand-vs-IR-pen tradeoff. Cheap, kid-safe, robust under projector lighting, multi-finger, and they're exactly what our custom YOLO is best at recognizing.

---

## 1. Goals & Non-Goals

### Goals
- A child walks up to the wall, sees a huge canvas, and draws within seconds.
- Latency from hand motion to projected ink ≤ 80 ms perceived.
- One-command launch; one-time calibration that persists.
- Works **with or without** the projector (project-onto-wall mode, or display-onto-screen mode).
- Works **with or without** the dot gloves (fallback: ArUco wand on a single camera).
- Floating HUD follows the player's active hand — no off-canvas menus.
- Local-only. No cloud.

### Non-Goals (this iteration)
- Multi-player simultaneous drawing.
- 3D / depth painting (Tilt-Brush style).
- Curved walls or projection mapping onto non-planar surfaces.
- Network sync, sharing UX, multi-page sketchbook.
- Quest controllers (already analyzed in source PRD3 and ruled out as fragile + macOS-hostile).
- Web / mobile rendering targets in MVP.

---

## 2. Hardware Inventory & Roles

| Device | Role | Notes |
|---|---|---|
| Yi camera @ 10.0.0.33 (yi-hack-v5) | Primary vision source A | RTSP + snapshot; PTZ supported |
| Yi camera @ 10.0.0.34 | Primary vision source B (stereo pair) | Same firmware family |
| Projector | Output display | If unavailable: render to attached screen instead |
| Mac (this laptop) | Compute / renderer host | Python backend + browser renderer |
| Custom YOLO weights | Dot-glove detection model | 6× lower memory, ~40% faster than baseline; subagent finding exact path |
| Dot gloves | Player wears one or two | High-contrast colored dots; one color per finger or per glove for ID |
| Android TV / TV stick (optional) | Side input via ADB | Only relevant if projector source is an Android TV; not on critical path |

### Latency hygiene for the Yi cameras
The cameras are RTSP/MJPEG over LAN. To keep glass-to-glass latency under control:
- Open with `cv2.CAP_FFMPEG` and `cv2.CAP_PROP_BUFFERSIZE=1`.
- Pass FFmpeg env: `OPENCV_FFMPEG_CAPTURE_OPTIONS=fflags;nobuffer|flags;low_delay|tune;zerolatency`.
- Each camera runs in its **own** capture thread, dropping all but the latest frame into a `Queue(maxsize=1)`.
- Inference loop pulls "newest only"; never blocks on capture.
- Prefer the **low-res stream** for the inference path (yi-hack-v5 exposes both); use high-res only for snapshots/debug.

(Source PRD2 details this aggressively. We follow it verbatim.)

---

## 3. Architecture

```
┌────────────────────────── WALL (projector or screen) ──────────────────────────┐
│                                                                                 │
│   Full-screen canvas (p5.js) — strokes + floating HUD that follows active hand  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                  ▲                                       ▲
                  │ HDMI                                  │ WebSocket (ws://localhost)
                  │                                       │
┌─────────────────┴───────────────────────────────────────┴────────────────────────┐
│                            Mac (macOS — primary)                                  │
│  ┌──────────────────────────────────────────┐    ┌──────────────────────────┐    │
│  │  projectart/  (Python 3.11)              │    │ renderer/  (browser)     │    │
│  │                                          │    │                          │    │
│  │   capture/                               │    │  index.html              │    │
│  │     yi_rtsp.py (per-cam thread, Q=1)     │    │  sketch.js (p5.js +      │    │
│  │   detection/                             │    │              p5.brush.js)│    │
│  │     yolo_dots.py  ← custom weights       │    │  hud.js (floating HUD    │    │
│  │     stereo.py     ← triangulation        │    │           that follows   │    │
│  │   geometry/                              │    │           the player)    │    │
│  │     wall_plane.py                        │    │  ws_client.js            │    │
│  │     homography.py                        │    │                          │    │
│  │     filter_one_euro.py                   │    └──────────────────────────┘    │
│  │   calibration/                           │                                    │
│  │     wizard_charuco.py                    │                                    │
│  │     wizard_4corner.py                    │                                    │
│  │     persist.py  → ~/.projectart/calib.json                                    │
│  │   server/                                │                                    │
│  │     ws.py  ← asyncio websockets          │                                    │
│  │     protocol.py                          │                                    │
│  │   inputs/                                │                                    │
│  │     gloves.py     ← primary (stereo)     │                                    │
│  │     wand.py       ← fallback (1 cam)     │                                    │
│  │     mouse.py      ← dev simulator        │                                    │
│  │   integrations/                          │                                    │
│  │     androidtv.py  ← optional, ADB        │                                    │
│  └──────────────────────────────────────────┘                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
                  ▲                ▲
                  │ RTSP (low-res) │ RTSP (low-res)
                  │                │
        Yi @ 10.0.0.33    Yi @ 10.0.0.34
```

### Per-frame data flow (target 30–60 Hz)

1. Each capture thread keeps the newest frame in its Q=1.
2. Inference loop pulls latest frames from both cams.
3. YOLO runs on each frame → dot detections `(class_id, cx, cy, conf)`.
4. Per-class correspondence between cams → triangulated 3D dot positions in camera-rig space.
5. **Player tracking**: median of dots → "active hand center"; HUD anchor.
6. **Wall plane intersect** + 2D homography → canvas pixels `(x, y)`.
7. **Contact gating**: dot is "in contact" if the wall-plane signed distance < ε (calibrated; default 1.5 cm).
8. **1€ filter** on canvas-space `(x, y)`.
9. WebSocket emit: `{type, x, y, contact, hud_anchor, gesture, velocity, ts}`.

The renderer treats every message as ground truth. No interpolation logic on the JS side that the Python side doesn't drive.

---

## 4. Input modes (pluggable)

`--input gloves|wand|mouse|androidtv` (defaults to `gloves` if both cameras are reachable, else `wand` if one camera, else `mouse`).

| Mode | Cameras | Markers | Contact source | Notes |
|---|---|---|---|---|
| `gloves` (primary) | 2 Yi | Custom dot pattern on gloves; YOLO classes per finger | Stereo triangulation + wall plane | Best UX |
| `wand` (fallback) | 1 Yi | Printed ArUco tag on a stick | Single-cam homography (no stereo); button on wand or BT clicker = trigger | Robust if YOLO/stereo flaky |
| `mouse` (dev) | none | mouse cursor | left-button = contact | For headless dev / CI |
| `androidtv` (side) | n/a | n/a | n/a | NOT a drawing input — pipes Android TV remote keys to the renderer for menu navigation when no glove user is present |

The pluggable layer matches drawmagic's design (`InputSource` Protocol). The contract:

```python
@dataclass
class PointerEvent:
    x: float                # canvas px
    y: float                # canvas px
    contact: bool
    hand_id: int            # 0 = right, 1 = left, etc.
    finger_id: int          # MediaPipe-style: 4 = thumb tip, 8 = index, ...
    confidence: float
    velocity: float         # px/s, for brush dynamics
    ts_ms: int
```

The HUD anchor is a separate event:

```python
@dataclass
class HudAnchorEvent:
    x: float                # canvas px — where to attach the floating menu
    y: float
    visible: bool           # false when player is out of frame
    ts_ms: int
```

---

## 5. The floating HUD

Specific to this design (not present in the source PRDs).

- **Anchor**: the median of detected glove dots, projected to canvas. Falls back to the YOLO `person` bounding-box center if no dots are visible.
- **Behavior**: A small ring of icons orbits the anchor, semi-transparent (~40% alpha). The ring lags the anchor by ~150 ms with a tween — it doesn't snap, so it feels like it's tagging along.
- **Tools shown**: brush, color, size, undo, clear (held), save, mode toggle.
- **Activation**:
  - **Hover-pinch**: pinch the active hand (thumb tip dot near index tip dot) for 350 ms → the closest tool icon pops to selected.
  - **Held-open-palm**: open palm → expand the ring into a radial menu.
- **Auto-hide**: after 4 s of no contact and no near-pinch, the ring fades to ~10% alpha so it doesn't dominate the canvas.

Why it's better than a corner menu: kids walk around. A corner menu means walking back to the corner. The HUD-follows-player pattern keeps tools always one short reach away.

---

## 6. Calibration

Two one-time wizards, both runnable in any order. Persist to `~/.projectart/calib.json`.

### 6.1 Camera intrinsics (per camera, ~30 s)
Standard ChArUco printed-board calibration. Each Yi camera is calibrated once in isolation. We need this for stereo triangulation to be metric.

### 6.2 Stereo extrinsics (~30 s)
Both cameras simultaneously observe a moving ChArUco board. Output: rotation + translation of cam B relative to cam A. We rectify on the fly so dot correspondence is constrained to a horizontal scanline.

### 6.3 Wall plane + projector homography (~30 s — only if projector present)
Projector displays a full-screen ChArUco. Both cameras detect it. We solve:
- The wall plane (3 points minimum; we use ~24 corner points) in cam-rig space.
- The 3×3 homography from wall-plane (u, v) → projector pixels.

### 6.4 4-corner glove poke (the fast path)
If the user wants to skip ChArUco geometry (e.g., the projector is mounted off-axis with keystone): tap each of the 4 projected corner targets with a glove dot. Solves a single planar homography end-to-end. Less accurate at corners than ChArUco but works in 30 s.

```json
// ~/.projectart/calib.json schema (draft)
{
  "version": 1,
  "camera_a": {
    "url": "rtsp://10.0.0.33/...",
    "K": [[..3x3..]],
    "dist": [k1, k2, p1, p2, k3]
  },
  "camera_b": { "url": "rtsp://10.0.0.34/...", "K": [...], "dist": [...] },
  "stereo": {
    "R_a_to_b": [[..3x3..]],
    "t_a_to_b": [..3..]
  },
  "wall_plane": {
    "normal":   [..3..],
    "centroid": [..3..],
    "uv_basis": { "u": [..3..], "v": [..3..] }
  },
  "homography_uv_to_canvas": [[..3x3..]],
  "canvas": { "w": 1920, "h": 1080 },
  "contact_epsilon_m": 0.015,
  "created_at": "2026-05-09T..."
}
```

---

## 7. Custom YOLO integration

The custom-trained weights (the ones with 6× memory savings + ~40% latency win) are the **primary** detector. Background subagent will report the exact path; for now the design assumes:

```python
# detection/yolo_dots.py
class DotDetector:
    def __init__(self, weights: Path):
        self.model = ultralytics.YOLO(str(weights))   # falls back to onnxruntime if .onnx
    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        # returns one Detection per dot:
        #   class_id (which finger / which glove), bbox center, conf
```

**Open question for tomorrow morning:** Was this model trained on dot gloves specifically, or on hands/COCO? If the latter, we have two options:
1. Use it as a `person`/`hand` detector + run a fast classical blob detector inside that ROI for the dots.
2. Fine-tune on a small dot-glove dataset over a coffee.

The subagent's report will resolve this.

If MediaPipe Hands ends up being needed for finger-skeleton gestures (pinch detection beyond just dot proximity), we run it **only inside the YOLO-cropped ROI** — same two-stage pattern PRD2 specifies, with the same latency benefit.

---

## 8. Renderer (browser, p5.js + p5.brush.js)

- Full-screen, borderless, on the secondary display (the projector).
- Black background. No window chrome. `cursor: none`.
- Brushes: `pen`, `marker`, `neon`, `rainbow`, `spray`, `stamp`. (drawmagic's taxonomy.)
- Stroke width modulated by `velocity` (faster → thinner) per ProjectorCanvas's recipe.
- Pressure: ignored in MVP (gloves don't have force sensors); controlled by velocity instead.
- Save canvas as PNG: `Cmd+S` operator hotkey → `~/.projectart/saves/YYYY-MM-DD-HHMMSS.png`.
- Undo: stroke-stack of length 50; HUD button.
- Clear: held-confirm 1.5 s; auto-snapshot before clear.

---

## 9. Optional: AndroidTV remote integration

Out of the critical path, but the user flagged it as potentially useful. The github reference `Jekso/AndroidTV-Remote-Controller` is a Python ADB wrapper for sending key codes to an Android TV.

**Where it fits:** if the projector source is an Android TV stick (Chromecast w/ Google TV, NVIDIA Shield, etc.), the AndroidTV is the device actually rendering our browser canvas. Then ADB control gives us:
- "Open the canvas page on launch" (`am start -a VIEW http://<mac-ip>:8000`)
- Volume / power / home / back from the Mac terminal — no IR remote required
- Potentially, "kid-safe lock" — when the canvas is up, ADB swallows key events so the kid can't browse to YouTube

**Status:** stub now (`integrations/androidtv.py` with a `is_androidtv_available()` probe + a key-passthrough); wire up only if it turns out the projector setup uses Android TV. Otherwise YAGNI.

---

## 10. Milestones & Acceptance

### MVP — minimum to call it shipped
- **M0** Repo scaffold; `pip install -e .`; `python -m projectart` boots browser canvas at black full-screen.
- **M1** Mouse input mode draws strokes on the canvas via WebSocket. (No camera yet — proves the renderer/contract.)
- **M2** Single Yi camera (10.0.0.33) → YOLO dot detection → 2D-projected canvas pixels using 4-corner calibration. Drawing works.
- **M3** Add second Yi camera, stereo triangulation, wall-plane contact gating. Draw quality and contact reliability noticeably improve.
- **M4** Floating HUD that follows the active hand with the brush/color/size/undo/clear ring. Pinch-to-select.
- **M5** ChArUco wizards (camera intrinsics, stereo extrinsics, wall plane) with persistence.
- **M6** Brush set (pen, marker, neon, rainbow), save-PNG, sound effects, undo.

### Nice-to-have
- N1 Wand fallback mode (`--input wand`).
- N2 Spray + stamp brushes.
- N3 AndroidTV ADB integration.
- N4 In-canvas video tutorial that auto-plays on first launch.
- N5 Multi-glove (player wears two; left = palette, right = brush, à la Tilt Brush).

### Acceptance criteria (gates for "done")
- **AC-1** Cold start to drawable canvas ≤ 90 s including first-time calibration.
- **AC-2** With both cameras + projector, ink lands within 2 cm of the dot at canvas center, standing 2 m from wall.
- **AC-3** Trigger (dot makes contact) → first ink pixel ≤ 80 ms (casual stopwatch).
- **AC-4** Five back-to-back drawings, no crashes, no calibration drift.
- **AC-5** Re-launching within 24 hrs skips calibration.
- **AC-6** With `--input mouse`, no cameras attached, the canvas still draws.
- **AC-7** With one Yi unreachable, app auto-degrades to single-cam wand mode and tells the user.

---

## 11. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | Yi RTSP latency too high (>200 ms) for interactive ink | Medium | Severe — kills the magic | Use low-res stream + nobuffer flags; if still bad, swap to USB capture or MJPEG over ZeroMQ as PRD2 specifies |
| R-2 | YOLO weights weren't trained on dot gloves | Medium | Slows MVP | Fall back to classical color-blob detection inside YOLO's `hand`/`person` ROI; or fine-tune overnight |
| R-3 | Front-projection shadows / occlusion break dot detection when player is between projector and wall | High | Major UX dip | Two-camera stereo (cameras are NOT co-axial with the projector) is exactly designed to avoid this; mount Yis off to the sides |
| R-4 | Calibration drift if projector is bumped | Medium | Looks broken | One-key recalibration (`R` from canvas) re-runs 4-corner glove poke in 30 s |
| R-5 | Player wanders out of frame | Medium | HUD floats off-canvas | Clamp HUD anchor to canvas bounds; auto-fade when no detection for 1 s |
| R-6 | YOLO inference too slow for 60 Hz on Mac without GPU | Medium | Visible jitter | The model is "6× less memory + ~40% faster" — should be fine on M-series CPU/Neural Engine, but profile early; downsample frames if needed |
| R-7 | Kid accidentally clears canvas | High | Sad child | 1.5 s hold + chime to confirm clear; auto-snapshot before clear (drawmagic's R-6) |
| R-8 | Bright projector hits a kid's eyes | High if ignored | Safety | Mount projector behind/above kid; clamp output brightness to ≤70%; document setup |

---

## 12. Tech stack (pinned)

```toml
# pyproject.toml (excerpt)
[project]
name = "projectart"
requires-python = ">=3.11"
dependencies = [
  "ultralytics>=8.3",
  "opencv-contrib-python==4.10.0.84",
  "numpy==1.26.4",
  "websockets==12.0",
  "imageio==2.34.1",
  "pydantic==2.7.0",
  # AndroidTV (optional):
  "adb-shell==0.4.4",
]
[project.optional-dependencies]
dev = ["pytest==8.2.0", "ruff==0.4.4", "rich"]
```

Renderer is a static `index.html` served from `python -m http.server` on port 8000 (or via the websocket server's static-file handler). p5.js + p5.brush.js loaded from CDN in MVP, vendored later.

---

## 13. Repo layout

```
projectart/
├── pyproject.toml
├── README.md
├── PRD.md                 → symlink or copy of this design doc
├── CLAUDE.md              # repo-level dev instructions
├── docs/
│   └── superpowers/specs/2026-05-09-projectart-design.md  ← this file
├── src/projectart/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── config.py
│   ├── capture/
│   │   └── yi_rtsp.py
│   ├── detection/
│   │   ├── yolo_dots.py
│   │   └── stereo.py
│   ├── geometry/
│   │   ├── wall_plane.py
│   │   ├── homography.py
│   │   └── filter_one_euro.py
│   ├── calibration/
│   │   ├── wizard_charuco.py
│   │   ├── wizard_4corner.py
│   │   └── persist.py
│   ├── server/
│   │   ├── ws.py
│   │   └── protocol.py
│   ├── inputs/
│   │   ├── base.py        # PointerEvent + HudAnchorEvent + InputSource Protocol
│   │   ├── gloves.py
│   │   ├── wand.py
│   │   └── mouse.py
│   └── integrations/
│       └── androidtv.py
├── renderer/
│   ├── index.html
│   ├── sketch.js
│   ├── hud.js
│   ├── brushes.js
│   └── ws_client.js
└── tests/
    ├── test_filter.py
    ├── test_homography.py
    ├── test_protocol.py
    └── test_stereo.py
```

---

## 14. Open questions for the morning

1. **YOLO weights**: which file, what classes, what training set? (Subagent finding now.)
2. **Projector availability**: confirmed mounted? Throw distance? Aspect ratio?
3. **Glove design**: does the user already have gloves with dots, or do we need to spec them? If we need to spec: recommended dot pattern below.
4. **Android TV**: is the projector source actually an Android TV? If not, drop §9 entirely.

### Recommended glove spec (if not yet built)
- Black or very-dark glove.
- Five dots per glove, one per finger tip, ~14 mm diameter, in five distinct **highly saturated** colors (one per finger): magenta (thumb), cyan (index), lime (middle), yellow (ring), orange (pinky).
- One additional dot on the back of the hand (white) — this is the HUD anchor anchor.
- Dots must be matte (no specular glints under projector light).
- High contrast against both wall and (importantly) against projected colors — the saturated set above is roughly perpendicular in HSV to most projected hues.

---

## 15. What this document is NOT

- It is not the implementation plan. The implementation plan comes next via `superpowers:writing-plans`.
- It does not pick the exact HUD icons / brush textures / sounds — those are tuned during M4–M6.
- It does not commit to GPU vs CPU inference until the YOLO weights are confirmed.

The implementation plan will reference this document and break each milestone into subagent-sized tasks.
