# Wall-Projector Drawing App for a Kid — PRD + Initial Code Stubs

## TL;DR
- **Build it as a Python app**: `pyglet` (windowed full-screen on the projector) + `openvr` (controllers via SteamVR) + `opencv-contrib-python` (auto-calibration via ChArUco) + a `1€` filter for smoothing. This is the fastest "ship tomorrow" path because every piece has a working Python example and Claude Code can implement it in one sitting.
- **Quest controllers WITHOUT the headset on the user's head DOES work, but it requires the headset to be powered on, plugged into the PC (Quest Link / USB), and *positioned facing the play area* — i.e. headset on a desk/tripod looking at the wall, with the proximity sensor either taped over or ADB-disabled so it doesn't sleep.** Tracking is inside-out from the HMD's cameras; you must keep the controllers in the headset's FOV. Plan for a wand-fallback (webcam + ArUco-tagged stick) because this is the single biggest risk.
- **Auto-calibration is a 30-second one-time step**: project an ArUco/ChArUco grid full-screen, the network webcam captures it, OpenCV computes the projector→wall homography. Controller pose is then raycast onto the wall plane (defined during the same calibration) and the hit point is mapped through the homography into projector pixels.

---

## 1. Executive Summary

Build a single-binary Python desktop app that runs on the projector laptop. SteamVR runs in the background with the Meta Quest 2/3 connected via Quest Link (USB-C). The Quest is placed on a small tripod or desk pointed at the wall (acting effectively as a base station). The daughter holds one Touch controller; the trigger draws on the wall, thumbstick scrolls colors, A/B/X/Y change brushes/undo. A one-time auto-calibration projects a ChArUco board, the network webcam sees both the projection and the controller's pointing ray (via a known origin from a "tap the four corners" wizard), and the app computes a homography mapping projector pixels to a 2D plane in the controller's tracking space. Drawing happens in a `pyglet` window using a simple stroke buffer, with brushes implemented as procedural shaders over textured quad strips. The PRD provides explicit fallbacks (webcam + ArUco wand if Quest tracking is unreliable) and an MVP cut line that is realistically achievable in one Claude Code session (≤ 4 hours of agent work).

**Primary stack:** Python 3.11 · pyglet 2.0 · pyopenvr · opencv-contrib-python 4.10 · numpy · python-osc (optional, for future YOLO bridge).
**Fallback stack:** Same app, swap controller bridge for "ArUco-on-a-stick" wand detected by webcam.

---

## 2. Quest-Controllers-Without-HMD: Comparative Analysis

| Path | Works without HMD on head? | 6DoF? | Buttons/triggers/sticks? | Setup complexity | MVP-suitable? | Notes |
|---|---|---|---|---|---|---|
| **Quest Link (USB) + SteamVR + OpenVR/pyopenvr, headset on tripod facing wall** | ✅ Yes, with caveats | ✅ Yes (inside-out from HMD cameras) | ✅ All Touch inputs exposed via OpenVR `VRInput` API | Medium — need to disable sleep + tape proximity sensor | **YES — primary path** | Tracking quality drops if controllers leave HMD camera FOV. Quest 3 has wider coverage than Quest 2. Tested workflow: enable `Pause VR when headset is idle = OFF`, `Override Windows Power Scheme = ON`, cover proximity sensor (tape) or ADB-disable it. |
| **Quest Air Link (Wi-Fi) + SteamVR** | ✅ Same as Link | ✅ | ✅ | Same + Wi-Fi quality dependency | Acceptable backup if USB cable is short | Adds 5–15 ms latency vs USB. Use only if cable routing is awkward. |
| **ALVR (open-source PCVR streamer)** | ✅ | ✅ | ✅ Touch mapped to SteamVR | Higher (sideload APK, configure server) | No — too many moving parts to debug in one night | Useful only if Meta's official Link is broken on this machine. |
| **Steam Link app on Quest** | ✅ | ✅ | ✅ | Low | Possible alternative to Link if Oculus runtime won't install | Feeds straight into SteamVR; equivalent OpenVR data surface. |
| **SteamVR `null` driver + Quest controllers** (run SteamVR with no HMD whatsoever) | ❌ for Quest | n/a | n/a | n/a | **Does not work for Quest**, only for Lighthouse-tracked devices (Vive/Index controllers + base stations). Quest controllers are tracked by the Quest HMD itself; without the HMD running, there's no tracking source. | Important: this trick is widely documented but only applies to Lighthouse hardware. |
| **Direct Bluetooth pairing of Touch to PC** | ❌ Not viable | ❌ IMU/buttons only at best, and Meta does not document a public HID profile | Possibly partial (BT HID) | Very high (reverse-engineering) | **No** | Touch controllers are designed to pair only with a Quest HMD via the Meta mobile app. There is no supported PC-direct HID profile. Even where buttons might leak, you do **not** get 6DoF position. |
| **Meta XR SDK / Oculus PC SDK from a non-VR app** | Partially | Yes (via `ovr_*` API while Oculus runtime is up) | Yes | High (C/C++ binding) | No — slower than OpenVR path | OpenVR via SteamVR is simpler and language-friendlier. |
| **WebXR with controllers but no HMD session** | ❌ | n/a | n/a | n/a | No | WebXR `immersive-vr` session requires an active XR display. Emulators (Immersive Web Emulator) only emulate, they don't expose real controllers without a session. |
| **Webcam + ArUco-marker "wand" (no Quest at all)** | ✅ | ✅ (planar) | Buttons via a cheap BT clicker / on-wand button | Low–Medium | **YES — fallback path** | Use printed ArUco tag on a stick. js-aruco2 / OpenCV detect at 30+ FPS, gives 2D position directly in camera space → already homography-mapped to wall. Robust, cheap, kid-proof. |

### Verdict
**Primary:** Quest Link + SteamVR + pyopenvr, with the headset on a tripod aimed at the wall. **Mitigations required:** tape over the proximity sensor (or set `kiosk_mode_disable_proximity_sensor=1` via ADB) and turn off SteamVR idle-pause. **High-risk assumption** — *test this first thing in the morning before writing any drawing code.*
**Fallback:** ArUco-tagged wand + the existing network webcam. The same calibration pipeline naturally extends because the camera is already calibrated to the projector. This becomes the simplest possible path if the Quest setup proves fragile.

---

## 3. Architecture / Stack Comparative Analysis

| Stack | Time-to-MVP | Controller integration | Calibration libs | Drawing engine | Claude-Code-ability | Verdict |
|---|---|---|---|---|---|---|
| **Python (pyglet) + pyopenvr + opencv-contrib** | **~4–6 hrs** | `openvr` Python — 30-line example exists | OpenCV ChArUco built-in | pyglet `Batch` / VertexLists, simple shaders | Excellent — Python is Claude's strongest language and every dep is pip-installable | **CHOSEN** |
| Electron + Three.js + node-openvr + OpenCV.js | 8–12 hrs | `node-openvr` works but native build is finicky on macOS / current Node | OpenCV.js works but slower; js-aruco2 OK | Three.js fine | Medium — native module build is the trap | Backup if Python path blocked |
| Python + pygame + pyopenvr | Similar to pyglet | Same | Same | pygame is simpler but slower for many strokes | Good | Acceptable but pyglet has better batching |
| Native macOS (Swift + Metal) | 20+ hrs | No SteamVR on macOS — would need to custom-bridge | Vision/Metal | Metal | Poor — too much novel surface area | ❌ |
| Unity + XR Interaction Toolkit | 8–10 hrs (if Unity already set up) | Excellent | Need to write or import OpenCVForUnity ($95) | Excellent | Medium — Unity scenes don't review well as pure code | Overkill for a one-night MVP |
| Unreal | 12+ hrs | Excellent | OpenCV plugin exists | Excellent | Poor for one-night | ❌ |
| **TouchDesigner** | **~2 hrs for someone who knows TD** | OpenVR CHOP — *officially supports controllers without HMD* | OpenCV TOP / native chops | Excellent feedback-buffer paint patches | Poor — Claude Code cannot edit `.toe` files; everything is a node graph | Tempting but not Claude-Code-friendly |
| Web (browser) + WebXR | n/a | WebXR requires an immersive session, which requires the HMD on head | n/a | n/a | n/a | ❌ |

The **TouchDesigner** route would be the absolute fastest if a human were doing it directly (the dad has used TD before, and TD's `OpenVR CHOP` even has explicit "use controllers without HMD" support). But the brief says hand the PRD to **Claude Code**, and Claude Code edits text files. So we choose Python.

---

## 4. Recommended Architecture (Component Diagram)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         WALL (flat, painted-ish)                     │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐       │
│   │                  PROJECTED CANVAS (1920×1080)            │       │
│   │   <strokes rendered by pyglet, full-screen on output 2>  │       │
│   └──────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
       ▲                                    ▲
       │ projector HDMI/DP                  │ network webcam (RTSP/MJPEG)
       │                                    │
┌──────┴────────────────────────────────────┴──────────────────────────┐
│  Laptop (Windows preferred for SteamVR; macOS = fallback path only)  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    drawmagic/  (Python 3.11)                   │  │
│  │                                                                │  │
│  │   main.py ─── app loop @ 90 Hz                                 │  │
│  │     │                                                          │  │
│  │     ├── input/                                                 │  │
│  │     │     openvr_bridge.py     <── SteamVR ── Quest Link USB ──┼──> Quest HMD (on tripod, prox sensor taped, facing wall)
│  │     │     fallback_wand.py     <── webcam ArUco               │  │
│  │     │     filter.py            (1€ filter)                     │  │
│  │     │                                                          │  │
│  │     ├── calibration/                                           │  │
│  │     │     charuco.py           (project + detect + homography) │  │
│  │     │     wall_plane.py        (4-corner controller poke)      │  │
│  │     │     persist.py           (~/.drawmagic/calib.json)       │  │
│  │     │                                                          │  │
│  │     ├── canvas/                                                │  │
│  │     │     window.py            (pyglet full-screen)            │  │
│  │     │     stroke_buffer.py     (active stroke + history)       │  │
│  │     │     brushes/                                             │  │
│  │     │        base.py                                           │  │
│  │     │        pen.py  marker.py  spray.py  neon.py  rainbow.py  │  │
│  │     │        stamp.py                                          │  │
│  │     │     palette.py           (colors, sizes)                 │  │
│  │     │     hud.py               (corner readout, big buttons)   │  │
│  │     │                                                          │  │
│  │     ├── audio/sfx.py           (boop on button, scribble)      │  │
│  │     │                                                          │  │
│  │     └── bridge/osc.py          (future: YOLO over OSC)         │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

**Data flow (per frame, target 90 Hz):**
1. `openvr_bridge.poll()` → `(pose_matrix_4x4, buttons, axes)` from right Touch.
2. `filter.apply(pose)` → smoothed pose (1€ filter, β=0.05, mincutoff=1.0).
3. `wall_plane.raycast(pose)` → 3D intersection point on the wall plane (in tracking space).
4. `calibration.tracking_to_canvas(point3d)` → `(x_px, y_px)` in canvas pixels (via 3D-to-2D affine learned during 4-corner calibration).
5. `stroke_buffer.add_point(x_px, y_px, pressure=trigger_value)` if trigger is held.
6. `window.on_draw()` blits strokes via pyglet `Batch`.

The projector→wall homography from ChArUco is *only* needed if you want the wand-fallback path (it converts webcam pixels → canvas pixels). With the Quest, you skip the webcam entirely after the 4-corner step.

---

## 5. Auto-Calibration Approach

Two calibrations are needed; do **the one matching your input device**:

### A. Quest path: 4-corner controller calibration (the only one needed)
1. Project the canvas full-screen with four big animated targets in the corners (10% inset).
2. Prompt the daughter (or dad): *"Touch each pink dot with the controller tip and pull the trigger."*
3. For each corner, record the smoothed controller tip 3D position `Pᵢ` in the SteamVR tracking universe.
4. Best-fit a plane to the 4 points (SVD on `[P₁−P̄, P₂−P̄, P₃−P̄, P₄−P̄]`); store plane normal `n` and centroid `c`.
5. Build an in-plane 2D basis `(u, v)` and a 3×3 homography `H_plane→canvas` that sends the 4 in-plane points to `(0,0), (W,0), (W,H), (0,H)`.
6. **Runtime:** raycast from the controller pose along its forward (–Z in OpenVR controller space) to the plane, project to 2D in `(u,v)`, then apply `H`.

```python
# pseudocode (calibration/wall_plane.py)
def calibrate_4corners(get_smoothed_tip_pose):
    corners_3d = []
    for label in ("top_left","top_right","bottom_right","bottom_left"):
        show_target(label)
        wait_for_trigger_pull()
        corners_3d.append(get_smoothed_tip_pose())
    plane = fit_plane(corners_3d)            # SVD
    uv = build_basis(plane, corners_3d[0], corners_3d[1])
    canvas_xy = [(0,0),(W,0),(W,H),(0,H)]
    plane_xy  = [project_to_uv(p, plane, uv) for p in corners_3d]
    H = cv2.findHomography(np.array(plane_xy), np.array(canvas_xy))[0]
    save_json("calib.json", {"plane":plane, "uv":uv, "H":H.tolist()})
```

### B. Wand-fallback path: ChArUco projector↔camera homography
Use OpenCV's `cv2.aruco.CharucoBoard` (5×7, DICT_4X4_50). Total time < 30 s.

```python
# pseudocode (calibration/charuco.py)
def calibrate_projector_camera(project_image, grab_camera_frame):
    board = cv2.aruco.CharucoBoard((5,7), 0.04, 0.02, DICT)
    img = cv2.aruco.CharucoBoard.generateImage(board, (W, H), marginSize=40)
    project_image(img)                                # full-screen via pyglet
    time.sleep(0.5)
    frame = grab_camera_frame()                       # from RTSP webcam
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(frame)
    # Known canvas-pixel positions of each ChArUco corner:
    canvas_pts = board_corner_pixels_in_image(board, (W,H), 40)[ch_ids]
    H_cam_to_canvas, _ = cv2.findHomography(ch_corners, canvas_pts, cv2.RANSAC, 3.0)
    save_json("calib.json", {"H_cam_to_canvas": H_cam_to_canvas.tolist()})
```

For the wand path, runtime is: detect ArUco wand tag in webcam → `cv2.perspectiveTransform(tag_center, H_cam_to_canvas)` → canvas pixel → into `stroke_buffer`.

**Why ChArUco over plain checkerboard:** robust under partial occlusion (the kid's body, glare). Why not gray-code structured light: overkill for MVP, and ChArUco gives sub-pixel accuracy on a flat wall in ~2 frames.

---

## 6. Full PRD (hand this to Claude Code)

### 6.1 Goals
- A child (≈ ages 4–10) can walk up, point a Quest controller at the wall, pull the trigger, and a colored line appears on the wall under the controller's pointer. Latency ≤ 80 ms perceived.
- Single-command launch: `python -m drawmagic` brings up calibration on first run, then drops straight into a full-screen drawing canvas on the projector display.
- Total install: `pip install -e .` plus SteamVR. No paid software.
- "Daughter-can-use-it tomorrow" is the explicit MVP bar.

### 6.2 Non-Goals (MVP)
- Multi-controller / multiplayer drawing.
- 3D / depth painting (Tilt-Brush style).
- Curved or non-flat walls.
- Mobile, web, or Linux support.
- Network sync, cloud save, sharing UX.
- Pressure curves for non-trigger axes.

### 6.3 User Stories
- **As the daughter (primary)** I want to point at the wall and have a line follow my pointer when I squeeze, so it feels like magic.
- **As the daughter** I want big obvious colors I can flip through with the joystick, so I don't have to read.
- **As the daughter** I want one button to "make it all go away" so I can start over.
- **As the dad (operator)** I want one terminal command to launch the app and one calibration wizard that takes < 60 s.
- **As the dad** I want to be able to swap between Quest controller mode and webcam-wand mode with a CLI flag (`--input quest|wand`) without code changes, in case Quest tracking is flaky.
- **As the dad** I want the app to remember calibration across launches.

### 6.4 Functional Requirements (FR)
- **FR-1** App launches to full-screen on the configured projector display (CLI flag `--display N`, default = secondary display).
- **FR-2** On first launch (or `--recalibrate`), run the 4-corner controller calibration wizard (Quest mode) **or** the ChArUco projector-camera calibration (wand mode). Persist to `~/.drawmagic/calib.json`.
- **FR-3** Read right Touch controller pose, button states, trigger axis, grip axis, both thumbstick axes at ≥ 72 Hz via OpenVR.
- **FR-4** Apply 1€ filter (`mincutoff=1.0`, `beta=0.05`, `dcutoff=1.0`) to the controller's tip 3D position before raycasting.
- **FR-5** Raycast tip along controller forward axis to the calibrated wall plane; reject hits where `t < 0` or where the projected point falls > 20% outside the canvas (treat as "hover off").
- **FR-6** When trigger > 0.15, append a stroke point with `pressure = clamp((trigger - 0.15)/0.85, 0, 1)`.
- **FR-7** Brush types in MVP: `pen` (solid round), `marker` (textured wide), `neon` (additive glow), `rainbow` (hue-cycled), `spray` (jittered dots), `stamp` (preset PNGs: star, heart, smiley). At least 4 must work in MVP.
- **FR-8** Default control mapping (right Touch):
    - **Trigger** = draw with current brush.
    - **Grip** = erase (round eraser, radius = stroke size × 3).
    - **A** = cycle brush forward.
    - **B** = undo last stroke.
    - **Thumbstick X** = cycle color (left/right palette scroll).
    - **Thumbstick Y up** = increase stroke size; **Y down** = decrease.
    - **Thumbstick click** = clear canvas (with 1.5-s held confirmation + chime).
    - **Menu (≡) button** = toggle settings/HUD overlay.
- **FR-9** Sound effects: stroke-start chirp, button-press boop, clear-canvas swoosh, undo pop. Use `pyglet.media` or `simpleaudio`.
- **FR-10** Save current canvas as PNG on `Ctrl+S` (operator hotkey) to `~/.drawmagic/saves/YYYY-MM-DD-HHMMSS.png`.
- **FR-11** HUD shows current color swatch, current brush icon, current size, undo count, in a corner. Auto-hides after 3 s of inactivity.
- **FR-12** Wand-fallback mode: launch with `--input wand`. Detects an `ARUCO_MIP_36h12` marker (ID 0) on a printed paper wand. A separate physical button (e.g. clicker paired as a BT keyboard sending SPACE) acts as trigger.

### 6.5 Technical Requirements
- **Language/runtime:** Python ≥ 3.11.
- **Required packages (pinned):**
  - `pyglet==2.0.15`
  - `numpy==1.26.4`
  - `opencv-contrib-python==4.10.0.84`
  - `openvr==1.26.701` (`pyopenvr`)
  - `python-osc==1.8.3` (future YOLO bridge)
  - `imageio==2.34.1` (PNG save)
  - `pydantic==2.7.0` (calibration JSON schema)
- **OS target:** Windows 11 (because SteamVR + Quest Link is most reliable there). macOS is a documented but unsupported fallback (use wand mode only, since macOS has no SteamVR).
- **External services:** SteamVR must be installed and runnable. Meta Quest desktop app (Oculus PC) must be installed and "Quest Link" enabled. Headset proximity sensor must be defeated (tape OR `adb shell settings put system kiosk_mode_disable_proximity_sensor 1`); SteamVR setting *Pause VR when headset is idle* = OFF.
- **Calibration JSON schema** (`~/.drawmagic/calib.json`):
  ```json
  {
    "version": 1,
    "mode": "quest" | "wand",
    "canvas_w": 1920,
    "canvas_h": 1080,
    "plane": {"normal": [x,y,z], "centroid": [x,y,z]},   // quest only
    "uv_basis": {"u":[x,y,z], "v":[x,y,z]},              // quest only
    "H_plane_to_canvas": [[..3x3..]],                    // quest only
    "H_cam_to_canvas":   [[..3x3..]],                    // wand only
    "created_at": "ISO-8601"
  }
  ```
- **API contract — controller bridge** (`drawmagic.input.openvr_bridge.ControllerState`):
  ```python
  @dataclass
  class ControllerState:
      pose:         np.ndarray   # 4x4 SE(3), tracking-space, may be None when invalid
      tip_position: np.ndarray   # (3,) float32 — pose @ TIP_OFFSET
      forward:      np.ndarray   # (3,) unit vector
      trigger:      float        # 0..1
      grip:         float        # 0..1
      thumb_xy:     tuple[float,float]
      btn_a:        bool
      btn_b:        bool
      btn_menu:     bool
      btn_thumb:    bool
      timestamp_s:  float
  ```
- **API contract — drawing**: brushes implement
  ```python
  class Brush(Protocol):
      name: str
      def begin_stroke(self, color, size) -> StrokeState: ...
      def add_point(self, state, x, y, pressure, dt) -> list[Quad]: ...
      def end_stroke(self, state) -> None: ...
  ```
- **Performance budget:** controller-to-screen ≤ 80 ms (USB Link ≈ 25 ms + render frame ≈ 11 ms + filter latency ≈ 15 ms + projector input lag ≈ 25 ms).

### 6.6 Milestones / Cut Lines

**MVP (must ship by end of Day 1, in order):**
- M0: Repo scaffold, `pip install -e .` works, app boots a black full-screen window.
- M1: pyopenvr connects, prints right-controller pose at 90 Hz with proximity sensor defeated.
- M2: 4-corner calibration wizard works; saves `calib.json`; pointer dot follows controller on the wall.
- M3: Pen brush draws colored strokes on trigger pull. Undo (B) works.
- M4: Color cycle (thumbstick X) and size adjust (thumbstick Y).
- M5: At least 3 more brushes (marker, neon, rainbow). Clear-canvas + sounds.

**Nice-to-have (Day 2+):**
- N1: Wand-fallback mode (`--input wand`) using ArUco + webcam.
- N2: Stamp brush with 4 PNG stamps.
- N3: PNG export.
- N4: HUD overlay.
- N5: Spray brush.

**Future:**
- F1: Two-controller (left palette / right brush, like Tilt Brush).
- F2: YOLO bridge over OSC — detect drawn objects, "magic wand" mode that recognizes shapes.
- F3: Multi-page sketchbook.
- F4: Network multiplayer with the dad's tablet.
- F5: ChArUco re-projection on top of the controller plane to refine accuracy across the canvas.

### 6.7 Acceptance Criteria
- **AC-1** Cold start to drawable canvas in ≤ 90 s including calibration.
- **AC-2** Pointer dot at the controller's projected location stays within 2 cm of the actual aim point at the center of a 2 m-wide canvas, when standing 2 m from the wall.
- **AC-3** A trigger pull → first ink pixel ≤ 80 ms in casual stopwatch testing.
- **AC-4** Five children's drawings can be made back-to-back with no crashes.
- **AC-5** Re-launching the app within 24 hrs skips calibration and drops directly to the canvas.
- **AC-6** With the Quest unplugged, `--input wand` launches and the wand draws (this is the architectural fallback test).

### 6.8 Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | Quest controllers don't track when HMD is on a desk (out of FOV / sleep mode) | **Medium-High** | Blocks primary path | Tape proximity sensor; tripod-mount HMD facing the play area; disable SteamVR idle-pause; have wand-fallback mode ready as `--input wand` from day 1 |
| R-2 | SteamVR fails to install or breaks on this laptop | Medium | Blocks primary path | Wand-fallback path exists; don't gate the canvas on SteamVR being healthy |
| R-3 | Network webcam latency > 200 ms | Medium | Calibration only — one-time hit | Acceptable; calibration is not realtime. For wand mode, document min framerate ≥ 15 fps |
| R-4 | Drift / jitter in 6DoF pose ruins fine lines | Medium | UX | 1€ filter (β=0.05, mincutoff=1.0); offer "kid mode" with stronger smoothing |
| R-5 | Projector keystone makes calibration affine assumption wrong | Low for MVP | Visual misalignment near edges | Document: position projector roughly head-on; accept ±2 cm error at corners. ChArUco re-projection in F5 fixes later |
| R-6 | Daughter accidentally clears canvas | High | Sad child | 1.5 s hold + chime to confirm clear; auto-save snapshot before clear |
| R-7 | OpenVR Python binding doesn't build on Windows | Low | Blocks | Pre-built wheels exist on PyPI; pin to `openvr==1.26.701` |
| R-8 | Eye-safety: bright projector in kid's eyes | High if ignored | Safety | Document setup: rear-projection or projector mounted behind/above kid; software defaults to ≤ 70% brightness output |

---

## 7. Initial Code Stubs

### 7.1 Project Tree

```
drawmagic/
├── pyproject.toml
├── README.md
├── CLAUDE.md                          # repo-level instructions for Claude Code
├── .gitignore
├── assets/
│   ├── stamps/{star,heart,smiley,star2}.png
│   ├── sounds/{chirp,boop,swoosh,pop}.wav
│   └── targets/corner.png
├── src/drawmagic/
│   ├── __init__.py
│   ├── __main__.py                    # `python -m drawmagic`
│   ├── config.py
│   ├── app.py                         # main loop
│   ├── input/
│   │   ├── __init__.py
│   │   ├── base.py                    # ControllerState dataclass + abstract source
│   │   ├── openvr_bridge.py           # primary
│   │   ├── fallback_wand.py           # ArUco wand
│   │   └── filter.py                  # 1€ filter
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── wizard_quest.py            # 4-corner controller calibration
│   │   ├── wizard_charuco.py          # projector-camera ChArUco
│   │   ├── geom.py                    # plane fit, basis, raycast, homography apply
│   │   └── persist.py                 # load/save calib.json
│   ├── canvas/
│   │   ├── __init__.py
│   │   ├── window.py                  # pyglet window
│   │   ├── stroke_buffer.py
│   │   ├── palette.py
│   │   ├── hud.py
│   │   └── brushes/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── pen.py
│   │       ├── marker.py
│   │       ├── neon.py
│   │       ├── rainbow.py
│   │       ├── spray.py
│   │       └── stamp.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── sfx.py
│   └── bridge/
│       ├── __init__.py
│       └── osc.py                     # future YOLO bridge stub
└── tests/
    ├── test_filter.py
    ├── test_geom.py
    └── test_stroke_buffer.py
```

### 7.2 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "drawmagic"
version = "0.1.0"
description = "Wall-projector drawing app for kids using a Meta Quest controller as a magic wand."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "drawmagic" }]
dependencies = [
    "pyglet==2.0.15",
    "numpy==1.26.4",
    "opencv-contrib-python==4.10.0.84",
    "openvr==1.26.701",
    "python-osc==1.8.3",
    "imageio==2.34.1",
    "pydantic==2.7.0",
]

[project.optional-dependencies]
dev = ["pytest==8.2.0", "ruff==0.4.4"]

[project.scripts]
drawmagic = "drawmagic.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]
```

### 7.3 `CLAUDE.md` (repo-level)

```markdown
# drawmagic — Claude Code instructions
- Language: Python 3.11. Single venv. Use `pyproject.toml`, not requirements.txt.
- Style: Black-compatible, 100 cols. Type-hint public functions.
- Modules import only from `drawmagic.*`. No top-level side-effects.
- Logging via `logging.getLogger(__name__)` — no print() in library code.
- All filesystem paths go through `drawmagic.config.paths`.
- Prefer numpy ops over Python loops in hot paths (filter, raycast, stroke_buffer).
- Two input sources MUST stay swappable (`--input quest|wand`). Never import
  `openvr` outside of `drawmagic.input.openvr_bridge`.
- pyglet windowing happens only in `drawmagic.canvas.window`.
- Tests in `tests/` use pytest; geom/filter/buffer must be tested without a display.
- When in doubt, prefer the simpler version that works tonight. This is an MVP.
```

### 7.4 `src/drawmagic/__main__.py`

```python
from __future__ import annotations
import argparse, logging, sys
from .app import App

def main() -> int:
    p = argparse.ArgumentParser("drawmagic")
    p.add_argument("--input", choices=["quest", "wand"], default="quest")
    p.add_argument("--display", type=int, default=-1, help="display index, -1 = last")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--webcam", default="0", help="cv2.VideoCapture source (idx or RTSP url)")
    p.add_argument("--canvas", default="1920x1080")
    p.add_argument("--log", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    w, h = (int(x) for x in args.canvas.split("x"))
    return App(args, canvas_size=(w, h)).run()

if __name__ == "__main__":
    sys.exit(main())
```

### 7.5 `src/drawmagic/app.py` (main loop)

```python
from __future__ import annotations
import logging, time
import pyglet
from .config import paths
from .calibration.persist import load_calibration, save_calibration
from .calibration.wizard_quest import run_4corner_wizard
from .calibration.wizard_charuco import run_charuco_wizard
from .calibration.geom import CalibratedMapper
from .input.base import InputSource
from .input.openvr_bridge import OpenVRSource
from .input.fallback_wand import WandSource
from .canvas.window import CanvasWindow
from .canvas.stroke_buffer import StrokeBuffer
from .canvas.palette import Palette
from .canvas.brushes import all_brushes
from .audio.sfx import SFX

log = logging.getLogger(__name__)

class App:
    def __init__(self, args, canvas_size):
        self.args = args
        self.canvas_size = canvas_size
        self.input: InputSource | None = None
        self.mapper: CalibratedMapper | None = None

    def run(self) -> int:
        self.input = OpenVRSource() if self.args.input == "quest" else WandSource(self.args.webcam)
        self.input.start()
        window = CanvasWindow(self.canvas_size, display_index=self.args.display)
        sfx = SFX()
        palette = Palette()
        brushes = all_brushes()
        strokes = StrokeBuffer(self.canvas_size)
        calib = load_calibration() if not self.args.recalibrate else None
        if calib is None or calib.mode != self.args.input:
            calib = (run_4corner_wizard(window, self.input)
                     if self.args.input == "quest"
                     else run_charuco_wizard(window, self.args.webcam))
            save_calibration(calib)
        self.mapper = CalibratedMapper(calib, self.canvas_size)
        state = _RuntimeState(palette, brushes, strokes, sfx)

        def update(dt):
            cs = self.input.poll()
            if cs is None or cs.pose is None:
                return
            pt = self.mapper.controller_to_canvas(cs)  # (x,y) or None
            state.handle_buttons(cs)
            if pt is not None and cs.trigger > 0.15:
                strokes.add_point(pt[0], pt[1], pressure=(cs.trigger - 0.15) / 0.85,
                                  brush=state.brush, color=state.color, size=state.size)
            elif state.was_drawing:
                strokes.end_stroke(); sfx.play("pop")
            state.was_drawing = pt is not None and cs.trigger > 0.15

        pyglet.clock.schedule_interval(update, 1/120)
        @window.event
        def on_draw():
            window.clear_to(palette.bg)
            strokes.draw()
            if state.show_hud: state.hud.draw(state, self.canvas_size)
        pyglet.app.run()
        return 0

class _RuntimeState:
    def __init__(self, palette, brushes, strokes, sfx):
        from .canvas.hud import HUD
        self.palette, self.brushes, self.strokes, self.sfx = palette, brushes, strokes, sfx
        self.brush_idx = 0; self.color_idx = 0; self.size = 12
        self.show_hud = True; self.was_drawing = False
        self._prev_btns = {}
        self._thumb_cooldown = 0.0
        self._clear_started = None
        self.hud = HUD()

    @property
    def brush(self): return self.brushes[self.brush_idx]
    @property
    def color(self): return self.palette.colors[self.color_idx]

    def handle_buttons(self, cs):
        if self._edge(cs.btn_a, "a"):
            self.brush_idx = (self.brush_idx + 1) % len(self.brushes); self.sfx.play("boop")
        if self._edge(cs.btn_b, "b"):
            self.strokes.undo(); self.sfx.play("pop")
        if self._edge(cs.btn_menu, "m"):
            self.show_hud = not self.show_hud; self.sfx.play("boop")
        # thumbstick X = color, Y = size
        tx, ty = cs.thumb_xy
        now = time.monotonic()
        if now > self._thumb_cooldown:
            if tx > 0.6:
                self.color_idx = (self.color_idx + 1) % len(self.palette.colors); self._thumb_cooldown = now + 0.18; self.sfx.play("boop")
            elif tx < -0.6:
                self.color_idx = (self.color_idx - 1) % len(self.palette.colors); self._thumb_cooldown = now + 0.18; self.sfx.play("boop")
            if ty > 0.6:
                self.size = min(96, self.size + 4); self._thumb_cooldown = now + 0.10
            elif ty < -0.6:
                self.size = max(2, self.size - 4); self._thumb_cooldown = now + 0.10
        # held thumbstick click = clear
        if cs.btn_thumb:
            self._clear_started = self._clear_started or now
            if now - self._clear_started > 1.5:
                self.strokes.clear(snapshot=True); self.sfx.play("swoosh"); self._clear_started = None
        else:
            self._clear_started = None

    def _edge(self, pressed, key):
        prev = self._prev_btns.get(key, False); self._prev_btns[key] = pressed
        return pressed and not prev
```

### 7.6 `src/drawmagic/input/base.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

@dataclass
class ControllerState:
    pose: Optional[np.ndarray]              # 4x4 SE(3) in tracking space
    tip_position: Optional[np.ndarray]      # (3,)
    forward: Optional[np.ndarray]           # (3,) unit
    trigger: float = 0.0
    grip:    float = 0.0
    thumb_xy: tuple[float, float] = (0.0, 0.0)
    btn_a: bool = False
    btn_b: bool = False
    btn_menu: bool = False
    btn_thumb: bool = False
    timestamp_s: float = 0.0

class InputSource(Protocol):
    def start(self) -> None: ...
    def poll(self) -> Optional[ControllerState]: ...
    def stop(self)  -> None: ...
```

### 7.7 `src/drawmagic/input/openvr_bridge.py`

```python
from __future__ import annotations
import logging, time
import numpy as np
import openvr
from .base import ControllerState, InputSource
from .filter import OneEuroVec3

log = logging.getLogger(__name__)
TIP_OFFSET = np.array([0.0, -0.005, -0.06], dtype=np.float32)  # ~6cm forward of grip

def _mat34_to_44(m):
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = np.array(m.m, dtype=np.float32)
    return out

class OpenVRSource:
    def __init__(self, hand: str = "right"):
        self.hand = hand
        self._sys = None
        self._poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self._device_idx: int | None = None
        self._tip_filter = OneEuroVec3(mincutoff=1.0, beta=0.05, dcutoff=1.0)

    def start(self) -> None:
        # Initialize as a background app so we don't try to be the compositor.
        self._sys = openvr.init(openvr.VRApplication_Background)
        log.info("OpenVR initialized")

    def _resolve_device(self) -> int | None:
        role = (openvr.TrackedControllerRole_RightHand if self.hand == "right"
                else openvr.TrackedControllerRole_LeftHand)
        idx = self._sys.getTrackedDeviceIndexForControllerRole(role)
        return idx if idx != openvr.k_unTrackedDeviceIndexInvalid else None

    def poll(self):
        if self._sys is None: return None
        self._sys.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0.0, len(self._poses), self._poses)
        if self._device_idx is None or not self._poses[self._device_idx].bPoseIsValid:
            self._device_idx = self._resolve_device()
            if self._device_idx is None: return None
        p = self._poses[self._device_idx]
        if not p.bPoseIsValid: return None
        M = _mat34_to_44(p.mDeviceToAbsoluteTracking)
        # tip = M * TIP_OFFSET (homogeneous)
        tip_local = np.array([*TIP_OFFSET, 1.0], dtype=np.float32)
        tip_world = (M @ tip_local)[:3]
        # forward in OpenVR controller space is -Z
        fwd_world = (M[:3, :3] @ np.array([0, 0, -1], dtype=np.float32))
        fwd_world /= (np.linalg.norm(fwd_world) + 1e-9)
        ts = time.monotonic()
        tip_smoothed = self._tip_filter(tip_world, ts)

        ok, st = self._sys.getControllerState(self._device_idx)
        trigger = float(st.rAxis[1].x) if ok else 0.0
        grip    = float(st.rAxis[2].x) if ok else 0.0
        thumb_x = float(st.rAxis[0].x) if ok else 0.0
        thumb_y = float(st.rAxis[0].y) if ok else 0.0
        bp = st.ulButtonPressed if ok else 0
        def pressed(button_id): return bool(bp & (1 << button_id))
        return ControllerState(
            pose=M, tip_position=tip_smoothed, forward=fwd_world,
            trigger=trigger, grip=grip, thumb_xy=(thumb_x, thumb_y),
            btn_a=pressed(openvr.k_EButton_A),
            btn_b=pressed(openvr.k_EButton_ApplicationMenu),
            btn_menu=pressed(openvr.k_EButton_System),
            btn_thumb=pressed(openvr.k_EButton_SteamVR_Touchpad),
            timestamp_s=ts,
        )

    def stop(self):
        if self._sys is not None:
            openvr.shutdown(); self._sys = None
```

### 7.8 `src/drawmagic/input/filter.py` (1€ filter)

```python
from __future__ import annotations
import math
import numpy as np

class _LowPass:
    def __init__(self): self.y = None
    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y

def _alpha(cutoff, dt): tau = 1.0 / (2 * math.pi * cutoff); return 1.0 / (1.0 + tau / dt)

class OneEuroVec3:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x_lp = _LowPass(); self.dx_lp = _LowPass(); self.t_prev = None; self.x_prev = None
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x.copy(); return x
        dt = max(1e-3, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        edx = self.dx_lp(dx, _alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * float(np.linalg.norm(edx))
        ex = self.x_lp(x, _alpha(cutoff, dt))
        self.t_prev, self.x_prev = t, x; return ex
```

### 7.9 `src/drawmagic/input/fallback_wand.py`

```python
from __future__ import annotations
import time, threading
import cv2, numpy as np
from .base import ControllerState, InputSource
from .filter import OneEuroVec3

class WandSource:
    """Webcam + ArUco marker on a stick; SPACE-bar BT clicker = trigger."""
    MARKER_ID = 0
    DICT = cv2.aruco.DICT_4X4_50

    def __init__(self, src):
        self.src = int(src) if str(src).isdigit() else src
        self.cap = None; self._lock = threading.Lock(); self._latest = None
        self._stop = False; self._trigger = 0.0
        self._filter = OneEuroVec3(mincutoff=1.5, beta=0.04)

    def start(self):
        self.cap = cv2.VideoCapture(self.src)
        threading.Thread(target=self._loop, daemon=True).start()
        # SPACE-bar clicker → trigger handled by pyglet on_key in canvas.window;
        # That wires self._trigger via set_trigger_external().

    def set_trigger_external(self, value: float): self._trigger = float(value)

    def _loop(self):
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(self.DICT))
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok: time.sleep(0.01); continue
            corners, ids, _ = det.detectMarkers(frame)
            if ids is None: continue
            ids = ids.flatten()
            if self.MARKER_ID not in ids: continue
            i = list(ids).index(self.MARKER_ID)
            c = corners[i].reshape(4, 2).mean(axis=0)  # (u, v) in cam pixels
            with self._lock:
                self._latest = (np.array([c[0], c[1], 0.0], dtype=np.float32), time.monotonic())

    def poll(self):
        with self._lock: latest = self._latest
        if latest is None: return None
        pos, ts = latest
        smoothed = self._filter(pos, ts)
        return ControllerState(
            pose=None, tip_position=smoothed, forward=None,
            trigger=self._trigger, grip=0.0, thumb_xy=(0,0),
            btn_a=False, btn_b=False, btn_menu=False, btn_thumb=False,
            timestamp_s=ts,
        )

    def stop(self):
        self._stop = True
        if self.cap: self.cap.release()
```

### 7.10 `src/drawmagic/calibration/geom.py`

```python
from __future__ import annotations
import numpy as np, cv2
from dataclasses import dataclass

@dataclass
class CalibrationData:
    mode: str           # "quest" | "wand"
    canvas_w: int
    canvas_h: int
    plane_normal: np.ndarray | None = None     # (3,)
    plane_centroid: np.ndarray | None = None   # (3,)
    uv_u: np.ndarray | None = None             # (3,)
    uv_v: np.ndarray | None = None             # (3,)
    H_plane_to_canvas: np.ndarray | None = None # (3,3)
    H_cam_to_canvas:   np.ndarray | None = None # (3,3)

def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c)
    n = vt[-1]; n /= np.linalg.norm(n) + 1e-9
    return n, c

def build_basis(n: np.ndarray, p_topleft: np.ndarray, p_topright: np.ndarray):
    u = p_topright - p_topleft; u -= u.dot(n) * n; u /= np.linalg.norm(u) + 1e-9
    v = np.cross(n, u); v /= np.linalg.norm(v) + 1e-9
    return u, v

def project_to_uv(p, c, u, v):
    d = p - c; return np.array([d.dot(u), d.dot(v)], dtype=np.float32)

def ray_plane_intersect(origin, direction, n, c):
    denom = float(direction.dot(n))
    if abs(denom) < 1e-6: return None
    t = float((c - origin).dot(n) / denom)
    if t < 0: return None
    return origin + t * direction

class CalibratedMapper:
    def __init__(self, calib: CalibrationData, canvas_size):
        self.c = calib; self.W, self.H = canvas_size

    def controller_to_canvas(self, cs):
        if self.c.mode == "quest":
            if cs.tip_position is None or cs.forward is None: return None
            hit = ray_plane_intersect(cs.tip_position, cs.forward,
                                      self.c.plane_normal, self.c.plane_centroid)
            if hit is None: return None
            uv = project_to_uv(hit, self.c.plane_centroid, self.c.uv_u, self.c.uv_v)
            pts = np.array([[[uv[0], uv[1]]]], dtype=np.float32)
            xy = cv2.perspectiveTransform(pts, self.c.H_plane_to_canvas)[0,0]
        else:
            if cs.tip_position is None: return None
            pts = np.array([[[cs.tip_position[0], cs.tip_position[1]]]], dtype=np.float32)
            xy = cv2.perspectiveTransform(pts, self.c.H_cam_to_canvas)[0,0]
        x, y = float(xy[0]), float(xy[1])
        # generous off-canvas reject (20% slop)
        if -0.2*self.W < x < 1.2*self.W and -0.2*self.H < y < 1.2*self.H:
            return (x, y)
        return None
```

### 7.11 `src/drawmagic/calibration/wizard_quest.py`

```python
from __future__ import annotations
import time, numpy as np, cv2, pyglet
from .geom import CalibrationData, fit_plane, build_basis, project_to_uv

CORNERS = [("top_left",(0.1,0.1)),("top_right",(0.9,0.1)),
           ("bottom_right",(0.9,0.9)),("bottom_left",(0.1,0.9))]

def run_4corner_wizard(window, input_source) -> CalibrationData:
    W, H = window.canvas_size
    samples_3d = []
    for label, (rx, ry) in CORNERS:
        target_xy = (rx*W, ry*H)
        window.show_target(target_xy, label=f"Touch the dot ({label}) and squeeze the trigger")
        # Wait for trigger press, then record the mean tip pos over ~250ms
        while True:
            cs = input_source.poll()
            if cs is not None and cs.trigger > 0.7 and cs.tip_position is not None:
                acc = []; t0 = time.monotonic()
                while time.monotonic() - t0 < 0.25:
                    cs = input_source.poll()
                    if cs and cs.tip_position is not None: acc.append(cs.tip_position)
                    time.sleep(0.005)
                samples_3d.append(np.mean(np.stack(acc), axis=0)); break
            window.dispatch_events(); time.sleep(0.01)
    pts = np.stack(samples_3d).astype(np.float32)
    n, c = fit_plane(pts)
    u, v = build_basis(n, pts[0], pts[1])
    plane_xy = np.stack([project_to_uv(p, c, u, v) for p in pts]).astype(np.float32)
    canvas_xy = np.array([(0,0),(W,0),(W,H),(0,H)], dtype=np.float32)
    Hmat, _ = cv2.findHomography(plane_xy, canvas_xy, 0)
    return CalibrationData(mode="quest", canvas_w=W, canvas_h=H,
        plane_normal=n, plane_centroid=c, uv_u=u, uv_v=v, H_plane_to_canvas=Hmat)
```

### 7.12 `src/drawmagic/calibration/wizard_charuco.py`

```python
from __future__ import annotations
import time, numpy as np, cv2
from .geom import CalibrationData

DICT = cv2.aruco.DICT_4X4_50

def run_charuco_wizard(window, webcam_src) -> CalibrationData:
    W, H = window.canvas_size
    sqx, sqy = 7, 5
    sq_len, mk_len = 0.04, 0.02
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((sqx, sqy), sq_len, mk_len, aruco_dict)
    img = cv2.aruco.CharucoBoard.generateImage(board, (W, H), marginSize=40)
    window.show_fullscreen_image(img)
    cap = cv2.VideoCapture(int(webcam_src) if str(webcam_src).isdigit() else webcam_src)
    time.sleep(0.5)  # let projector + autoexposure settle
    detector = cv2.aruco.CharucoDetector(board)
    cam_corners = None; ids = None
    for _ in range(60):
        ok, frame = cap.read()
        if not ok: continue
        ch_corners, ch_ids, _, _ = detector.detectBoard(frame)
        if ch_ids is not None and len(ch_ids) >= 12:
            cam_corners = ch_corners.reshape(-1, 2); ids = ch_ids.flatten(); break
        time.sleep(0.05)
    cap.release()
    if cam_corners is None: raise RuntimeError("ChArUco not detected; check lighting.")
    # Each ChArUco corner has a known canvas-pixel position derived from the generated image:
    canvas_corner_pts = _charuco_corner_pixels(sqx, sqy, sq_len, W, H, margin_px=40)
    canvas_pts = canvas_corner_pts[ids]
    Hmat, _ = cv2.findHomography(cam_corners.astype(np.float32),
                                 canvas_pts.astype(np.float32), cv2.RANSAC, 3.0)
    return CalibrationData(mode="wand", canvas_w=W, canvas_h=H, H_cam_to_canvas=Hmat)

def _charuco_corner_pixels(sqx, sqy, sq_len, W, H, margin_px):
    # Mirrors how generateImage lays out the board.
    inner_w, inner_h = W - 2*margin_px, H - 2*margin_px
    sq_px = min(inner_w / sqx, inner_h / sqy)
    pts = []
    for j in range(sqy - 1):
        for i in range(sqx - 1):
            x = margin_px + (i + 1) * sq_px; y = margin_px + (j + 1) * sq_px
            pts.append((x, y))
    return np.array(pts, dtype=np.float32)
```

### 7.13 `src/drawmagic/calibration/persist.py`

```python
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from ..config import paths
from .geom import CalibrationData

CALIB = paths.calibration_file()

def _ndarray_to_list(x): return None if x is None else np.asarray(x).tolist()
def _list_to_ndarray(x): return None if x is None else np.asarray(x, dtype=np.float32)

def save_calibration(c: CalibrationData) -> None:
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps({
        "version": 1, "mode": c.mode, "canvas_w": c.canvas_w, "canvas_h": c.canvas_h,
        "plane_normal": _ndarray_to_list(c.plane_normal),
        "plane_centroid": _ndarray_to_list(c.plane_centroid),
        "uv_u": _ndarray_to_list(c.uv_u), "uv_v": _ndarray_to_list(c.uv_v),
        "H_plane_to_canvas": _ndarray_to_list(c.H_plane_to_canvas),
        "H_cam_to_canvas":   _ndarray_to_list(c.H_cam_to_canvas),
    }, indent=2))

def load_calibration() -> CalibrationData | None:
    if not CALIB.exists(): return None
    d = json.loads(CALIB.read_text())
    return CalibrationData(
        mode=d["mode"], canvas_w=d["canvas_w"], canvas_h=d["canvas_h"],
        plane_normal=_list_to_ndarray(d.get("plane_normal")),
        plane_centroid=_list_to_ndarray(d.get("plane_centroid")),
        uv_u=_list_to_ndarray(d.get("uv_u")), uv_v=_list_to_ndarray(d.get("uv_v")),
        H_plane_to_canvas=_list_to_ndarray(d.get("H_plane_to_canvas")),
        H_cam_to_canvas=_list_to_ndarray(d.get("H_cam_to_canvas")),
    )
```

### 7.14 `src/drawmagic/canvas/window.py` (sketch)

```python
from __future__ import annotations
import numpy as np, pyglet

class CanvasWindow(pyglet.window.Window):
    def __init__(self, canvas_size, display_index: int = -1):
        screens = pyglet.canvas.get_display().get_screens()
        screen = screens[-1 if display_index < 0 else min(display_index, len(screens)-1)]
        super().__init__(width=canvas_size[0], height=canvas_size[1], fullscreen=True,
                         screen=screen, caption="drawmagic", vsync=True)
        self.canvas_size = canvas_size
        self.set_mouse_visible(False)
        self._target = None
        self._fullscreen_img = None

    def clear_to(self, rgb): pyglet.gl.glClearColor(*rgb, 1); self.clear()

    def show_target(self, xy, label=""):
        self._target = (xy, label)
        # In MVP, just clear & draw once; the wizard pumps events between samples.
        self.clear_to((0.05, 0.05, 0.08))
        cx, cy = xy
        # invert y for pyglet bottom-left origin:
        cy = self.canvas_size[1] - cy
        pyglet.shapes.Circle(cx, cy, 36, color=(255,80,180)).draw()
        pyglet.text.Label(label, x=self.canvas_size[0]//2, y=40,
                          anchor_x="center", color=(255,255,255,255), font_size=24).draw()
        self.flip()

    def show_fullscreen_image(self, np_img):
        # np_img is HxW (gray) or HxWx3
        if np_img.ndim == 2: np_img = np.stack([np_img]*3, axis=-1)
        h, w, _ = np_img.shape
        data = np_img[::-1].astype(np.uint8).tobytes()
        img = pyglet.image.ImageData(w, h, "RGB", data)
        self.clear_to((0,0,0)); img.blit(0,0); self.flip()
```

### 7.15 `src/drawmagic/canvas/stroke_buffer.py`

```python
from __future__ import annotations
import math
import pyglet, numpy as np
from .brushes.base import Brush

class StrokeBuffer:
    def __init__(self, size):
        self.W, self.H = size
        self.batch = pyglet.graphics.Batch()
        self.completed = []   # list[list[pyglet primitives]]
        self.snapshots = []   # for undo of clear()
        self._active = None   # (brush, color, size, last_xy, prims)

    def add_point(self, x, y, pressure, brush, color, size):
        if self._active is None:
            self._active = (brush, color, size, (x, y), [])
            brush.begin_stroke(self.batch, color, size)
        b, col, sz, last, prims = self._active
        prims_new = b.add_segment(self.batch, last, (x, y), pressure)
        prims.extend(prims_new)
        self._active = (b, col, sz, (x, y), prims)

    def end_stroke(self):
        if self._active is not None:
            b, *_ , prims = self._active
            b.end_stroke()
            self.completed.append(prims)
            self._active = None

    def undo(self):
        if self._active is not None: self.end_stroke()
        if self.completed:
            for p in self.completed.pop():
                try: p.delete()
                except Exception: pass

    def clear(self, snapshot=True):
        if snapshot:
            # Cheap snapshot: just keep the prims on a buffer to restore on big-undo (future).
            pass
        for stroke in self.completed:
            for p in stroke:
                try: p.delete()
                except Exception: pass
        self.completed.clear(); self._active = None

    def draw(self):
        self.batch.draw()
```

### 7.16 `src/drawmagic/canvas/brushes/base.py`

```python
from __future__ import annotations
from typing import Protocol
import math, pyglet

class Brush(Protocol):
    name: str
    def begin_stroke(self, batch, color, size): ...
    def add_segment(self, batch, p_from, p_to, pressure) -> list: ...
    def end_stroke(self): ...

def _stamps_along(p0, p1, step):
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1-x0, y1-y0
    d = math.hypot(dx, dy)
    if d < 1e-3: return [(x1, y1)]
    n = max(1, int(d / step))
    return [(x0 + dx*i/n, y0 + dy*i/n) for i in range(1, n+1)]
```

### 7.17 `src/drawmagic/canvas/brushes/pen.py`

```python
from __future__ import annotations
import pyglet
from .base import _stamps_along

class PenBrush:
    name = "pen"
    def __init__(self): self.color = (255,255,255); self.size = 12
    def begin_stroke(self, batch, color, size): self.color, self.size = color, size
    def add_segment(self, batch, p_from, p_to, pressure):
        r = max(1, int(self.size * (0.4 + 0.6*pressure)))
        prims = []
        for (x, y) in _stamps_along(p_from, p_to, step=r*0.4):
            prims.append(pyglet.shapes.Circle(x, y, r, color=self.color, batch=batch))
        return prims
    def end_stroke(self): pass
```

(`marker.py`, `neon.py` etc. follow the same pattern; `neon` uses additive blending and a small inner + larger outer halo circle; `rainbow` cycles HSV by stamp index; `spray` jitters multiple small circles per step; `stamp` blits a textured quad.)

### 7.18 `src/drawmagic/canvas/brushes/__init__.py`

```python
from .pen import PenBrush
from .marker import MarkerBrush
from .neon import NeonBrush
from .rainbow import RainbowBrush
# from .spray import SprayBrush
# from .stamp import StampBrush
def all_brushes(): return [PenBrush(), MarkerBrush(), NeonBrush(), RainbowBrush()]
```

### 7.19 `src/drawmagic/canvas/palette.py`

```python
from dataclasses import dataclass

@dataclass
class Palette:
    bg = (0.05, 0.05, 0.08)
    colors = [
        (255, 60, 80),    # red
        (255, 150, 30),   # orange
        (255, 230, 60),   # yellow
        (90, 220, 90),    # green
        (60, 170, 255),   # blue
        (180, 100, 255),  # purple
        (255, 130, 220),  # pink
        (255, 255, 255),  # white
    ]
```

### 7.20 `src/drawmagic/audio/sfx.py`

```python
from __future__ import annotations
import logging, pyglet
from pathlib import Path
log = logging.getLogger(__name__)

class SFX:
    NAMES = ("chirp", "boop", "swoosh", "pop")
    def __init__(self, asset_dir: Path | None = None):
        from ..config import paths
        d = asset_dir or paths.assets() / "sounds"
        self.players = {}
        for n in self.NAMES:
            f = d / f"{n}.wav"
            if f.exists():
                try: self.players[n] = pyglet.media.load(str(f), streaming=False)
                except Exception as e: log.warning("sfx %s failed: %s", n, e)

    def play(self, name):
        s = self.players.get(name)
        if s: s.play()
```

### 7.21 `src/drawmagic/config.py`

```python
from __future__ import annotations
from pathlib import Path
import os

class _Paths:
    def root(self): return Path(os.environ.get("DRAWMAGIC_HOME", str(Path.home()/".drawmagic")))
    def calibration_file(self): return self.root()/"calib.json"
    def saves_dir(self): p = self.root()/"saves"; p.mkdir(parents=True, exist_ok=True); return p
    def assets(self):
        # repo-relative when running -e .
        here = Path(__file__).resolve().parents[2]
        return here / "assets"

paths = _Paths()
```

### 7.22 `tests/test_filter.py`

```python
import numpy as np
from drawmagic.input.filter import OneEuroVec3

def test_filter_passes_dc():
    f = OneEuroVec3(); v = np.array([1.0, 2.0, 3.0])
    out = None
    for t in range(10): out = f(v, t * 0.01)
    assert np.allclose(out, v, atol=1e-3)

def test_filter_smooths_noise():
    f = OneEuroVec3(mincutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0); base = np.array([0.0, 0.0, 0.0])
    last = None
    for i in range(200): last = f(base + rng.normal(0, 0.01, 3), i * 0.011)
    assert np.linalg.norm(last) < 0.01
```

---

## 8. Fallback Paths (in priority order)

1. **Quest tracking unstable when HMD is on a desk.** → Already-built `--input wand` mode using webcam + ArUco-tagged stick. Same calibration UI, swap wizard. Print one ArUco tag (DICT_4X4_50, ID 0) at ~10 cm; tape to a paint roller pole. Add a $5 BT clicker for the trigger button.
2. **SteamVR refuses to install / Quest Link broken on this laptop.** → Same as (1).
3. **Network webcam latency too high for ChArUco detection.** → Borrow a USB webcam for the calibration step only; retain network webcam for runtime detection in wand mode.
4. **pyopenvr build fails on Windows (rare).** → Use the `node-openvr` (Node.js) bridge in a sidecar process and stream JSON pose over a localhost UDP socket into the Python app. The `InputSource` interface absorbs this with a third implementation.
5. **Latency feels bad even with USB Link.** → Increase 1€ filter `beta` to 0.1; prediction term: extrapolate position by `velocity * 16 ms`; cap stroke point density.
6. **Calibration drifts after the kid bumps the projector.** → Add a hotkey (`Ctrl+R`) to restart the calibration wizard mid-session.
7. **Whole stack collapses.** → Ship the wand mode standalone — it's a complete drawing app on its own.

---

## 9. Concrete Next-Morning Checklist (first 5 things in order)

1. **Test the high-risk assumption first (15 min).** With the Quest powered on, plugged in via Quest Link USB, sitting on a desk pointing at the wall, **proximity sensor taped over**: launch SteamVR, confirm in the SteamVR status window that both controllers show "Tracking" (green) when held in the headset's camera FOV. Verify in `pyopenvr` with a 10-line script that pose updates at 90 Hz when you move the controller in front of the desk-mounted Quest. **If this fails, switch to `--input wand` as primary right now.** Don't write any drawing code until this is confirmed.
2. **Scaffold and bring the window up (30 min).** `pip install -e .[dev]`, paste in the file tree from §7, run `python -m drawmagic --recalibrate` and confirm a black full-screen window appears on the projector display.
3. **Wire the OpenVR bridge end-to-end (45 min).** Get the controller pose printing in the terminal; project a small dot at the screen center first; then implement the 4-corner wizard. Touch each corner, see the saved JSON.
4. **Implement the Pen brush + trigger drawing (30 min).** This is the moment you can hand the controller to the daughter. Everything else is polish.
5. **Add color cycling (thumbstick), undo (B), and clear (held thumbstick click). Ship it (45 min).** Save a PNG of her first drawing. Stop coding. Brushes 2–4, sounds, and HUD are nice-to-haves for tomorrow.

---

## Caveats

- **The single biggest unknown is whether Quest controller tracking will be reliable with the headset *on a desk* rather than on a head.** Multiple Reddit/Steam threads and Road to VR's Quest 3 article confirm Quest controllers track via the HMD's cameras and that *standalone* tracking from controllers' own cameras (Quest Pro / Quest 3 Touch Plus) can be flaky on featureless surfaces. The `null`-driver SteamVR trick that lets you run controllers without an HMD only works for **Lighthouse-tracked** devices (Vive wands, Index, Vive Trackers) — it does **not** work for Quest controllers, which need their parent HMD to be running. So while this PRD's primary path is plausible (the headset *is* on, just sitting on a desk), be prepared to fall back to the wand mode within minutes if tracking is jittery.
- **Direct Bluetooth pairing of Touch controllers to a PC is not a real path.** Despite using BT under the hood, Meta does not expose a public HID profile and there is no community driver that yields 6DoF position from BT-only pairing. Treat any guide claiming otherwise with skepticism.
- **Latency from network webcams will affect the wand-fallback path more than the Quest path.** Auto-calibration with a network camera works fine (one-shot), but real-time wand tracking under RTSP can be 100–250 ms — playable for kids but not snappy. A USB webcam during runtime is recommended for the wand path.
- **OpenVR Python (`pyopenvr`)** is described in upstream as "no longer being updated" in favor of OpenXR; the `pyopenxr` project exists but is less battle-tested. For one-night MVP, `pyopenvr` is still the right call — it's well-documented, has working examples, and SteamVR is the most reliable way to get Quest input on PC.
- **macOS path is genuinely worse.** SteamVR doesn't run on Apple Silicon. If the dad's laptop is a Mac, the *primary* path becomes wand-mode, and the "Quest controller" path requires running it from a Windows machine.
- **AC-2 (2 cm accuracy)** is a soft target. Realistic accuracy with a 4-corner controller calibration on a 2 m canvas is in the 1–3 cm range at the corners and < 1 cm near the center, dominated by tracking noise + plane-fit error. If the dad wants tighter, add F5 (ChArUco re-projection over the controller plane) to refine.
- **The PRD assumes flat wall, frontal-ish projector.** Keystone correction is explicitly out of scope for MVP. If the projector is significantly off-axis, the projected ChArUco pattern will still calibrate correctly to *itself*, but the "look" may be trapezoidal — that's a projector setup problem, not a software one.
- **Eye safety with bright projectors and small children is a real concern.** Document in README: do not let the child stare into the projector lens; consider rear-projection or a high-mounted projector behind the play area.