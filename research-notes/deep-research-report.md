# Interactive Wall Drawing System for Your Daughter

## Executive recommendation

The best path is **not** one single path. It is a two-step plan. First, get something delightful working fast with an **IR stylus** and a projector, because that skips the hardest problem: reliably knowing when a hand is actually **touching the wall** instead of just hovering in front of it. The cleanest off-the-shelf version of that is entity["software","Refurboard","camera-based interactive whiteboard software"] plus entity["software","OpenBoard","interactive whiteboard software"]. urlRefurboardturn30view0 already turns a projector or display into an interactive whiteboard using a webcam and IR pen on macOS, Windows, and Linux, and urlOpenBoardturn32view0 is cross-platform whiteboard software built for interactive boards and projector-classroom setups. citeturn30view0turn31view0turn32view0

Then, build the **custom version you actually want**: a browser or Electron app with your fast YOLO detector for stylus or fingertip tracking, entity["software","MediaPipe","Google AI Edge on-device ML framework"] for hand landmarks and gestures, and entity["software","OpenCV","open-source computer vision library"] for calibration and wall-plane mapping. That gives you the fun layer you described: brush changes, stroke modes, color switching, menus, hover effects, and eventually true hand drawing without needing to hijack some random paint program like a raccoon hot-wiring a Tesla. citeturn14view0turn14view1turn14view2turn14view3turn16view0turn16view1turn16view3

I would **not** make the initial implementation controller-first. Official Quest PC-VR linking is documented around a **Windows** computer, while Macs are documented under Remote Desktop rather than the main Horizon Link PC VR flow; meanwhile, entity["software","Open Brush","virtual reality painting software"] explicitly says Mac has no currently supported VR headset path and only limited desktop or monoscopic support. In plain English: Quest controllers are a decent side quest, not the main quest, if you want this on macOS. citeturn9view4turn9view5turn26search2turn26search13

## Existing tools that can get you there faster

The fastest zero-code or near-zero-code stack is still **IR pen + Refurboard + an existing whiteboard app**. Refurboard’s own technical guide says it uses OpenCV-based IR blob detection, adaptive brightness gating, and a 3×3 homography to map camera coordinates to screen coordinates, then drives the system cursor on macOS, Windows, and Linux. In other words, most of the boring plumbing is already done. You just calibrate, touch the wall with an IR pen, and the computer thinks it is a mouse. That is boring in the best possible way. citeturn31view0turn30view0

If you want commercial interactive-projection stacks instead of building your own, the two strongest “already made a lot of this problem go away” options are urlLUMOplayturn4search0 and urlMotioncubeturn34view1. entity["software","LUMOplay","interactive floor and wall projection software"] supports webcams as well as 3D cameras, has calibration tools for both, supports multi-camera seaming, and exposes a Unity SDK for custom apps. entity["software","Motioncube Player","interactive projection software"] supports depth cameras, IR cameras, IR pens, touch devices, and free downloadable player software, but its current wall-first templates lean more toward games and interactions than a dedicated freeform paint workflow. These products are great if you want a fast, polished install base, but you will still outgrow them if the goal is a custom kid-friendly drawing experience with your own gesture semantics. citeturn13view1turn13view2turn33view0turn34view0turn34view1

For the projection-mapping side of the house, entity["software","TouchDesigner","real-time visual development platform"] and entity["software","MadMapper","projection mapping software"] are the serious tools worth knowing. TouchDesigner ships projection mapping tools such as Projection Mapping, Kantan Mapper, and camSchnappr, and its non-commercial license is free for personal or learning use. MadMapper supports camera input on macOS and exposes live control through MIDI and OSC. Both are excellent if you want polished visual effects, masks, mapping fixes, and show-control behavior around the drawing app; neither replaces the need for a real tracker and input model. Think of them as the stage crew, not the child holding the crayon. urlTouchDesignerturn19search0 urlMadMapperturn6search2 citeturn36view0turn36view1turn36view2turn36view3turn9view0turn9view1turn9view2

If you are willing to buy dedicated hand-tracking hardware, entity["software","TouchFree","touchless interaction software"] and the official urlUltraleap Hyperion docsturn39view1 are the most relevant prebuilt pieces. TouchFree is designed to sit invisibly on top of an existing interface and convert hand movements into positional cursor data, but its own docs say it is **Windows-only**. Hyperion, by contrast, is available for Windows, macOS, and Android, includes a tracking SDK, and adds direct access to the IR camera hardware plus marker tracking. So if you want premium hand tracking while staying custom on Mac, Hyperion is viable; if you want turnkey touchless cursor control over an ordinary app, TouchFree is viable only if you are willing to go Windows. urlTouchFree docsturn39view0 citeturn39view0turn39view1turn39view3

## The best custom architecture

The most sensible custom architecture is a **Python tracking backend** plus a **full-screen browser or Electron renderer**. The tracking backend should own cameras, calibration, detection, gesture recognition, and contact logic. The renderer should own brush behavior, layers, history, saving, and the giant child-sized UI. This division is nice because your current YOLO work lives happily in Python, while the front end stays easy to theme and dead simple to deploy on macOS. citeturn11search0turn11search1turn14view4turn35view0turn35view1

For hand analysis, let MediaPipe do what MediaPipe is good at. The official Gesture Recognizer returns recognized hand gestures plus hand landmarks in real time, and the Hand Landmarker returns image-coordinate landmarks, world-coordinate landmarks, and handedness from live streams. MediaPipe Tasks documentation also explicitly positions the stack as optimized for real-time on-device pipelines. That makes it a better fit for gesture semantics than trying to force your detector to become a whole-body interaction stack. My recommendation is straightforward: **YOLO detects the thing you customized it for**—stylus tip, fingertip marker, brush proxy, or other bespoke objects—while MediaPipe handles **hands and gestures**. citeturn14view0turn14view1turn14view2turn14view3turn14view5

For geometry, use OpenCV end to end. The calibration docs state that camera calibration produces intrinsics and distortion coefficients and that they generally only need to be done once unless the optics change. OpenCV also documents ChArUco- and ArUco-based calibration, homography for planar mapping between two planes, and PnP pose estimation from 3D-2D correspondences. In practice, that means: print a ChArUco board once, calibrate each camera once, solve the wall plane, compute the homography into projector pixels, save the profile, and stop reliving calibration trauma every session. citeturn16view0turn16view1turn16view2turn16view3

For the drawing surface, a browser or Electron front end is not a compromise. The canvas APIs are mature, and the documentation for OffscreenCanvas explicitly says rendering can run inside a worker to keep heavy canvas work off the main thread. The same documentation set also includes a canvas drawing example built around pointer events. That means you can render an always-fullscreen wall canvas, keep the UI responsive, and accept position updates from your Python tracker over a localhost WebSocket. It is a very practical stack, not a toy stack. citeturn35view0turn35view1

The real design decision is **what starts a stroke**. My recommendation is to support three input modes, in this order. **P0:** IR pen or high-contrast marker, because it works immediately and gives you reliable “contact.” **P1:** hand gestures only for tool switching, menus, undo, and mode changes. **P2:** true camera-only hand drawing, but only with either a second side camera or a depth sensor so you can distinguish hover from contact. Trying to infer “touching the wall” from one laggy monocular stream is where good intentions go to die. Front projection already creates occlusion and shadow problems; the McGill projector-camera work documents that front projection creates shadows when a person stands between projector and display, which is exactly what your daughter will do because she is, very reasonably, expected to stand near the wall she is drawing on. citeturn18view1turn14view1turn16view1

## Hardware choices and the traps that matter

With the hardware you already have, the most important improvement is **camera path quality**, not model cleverness. If those webcams are being piped as network streams, replace that with direct USB/UVC capture if possible. Refurboard’s own setup advice recommends USB instead of Wi‑Fi when using a phone as a webcam “for the most responsive experience,” which is the polite version of saying latency is a mood-killer. Also make sure the camera sees the full projected area, the mount is rigid, and all four corners are visible during calibration. citeturn30view0

If you are willing to buy **one cheap thing**, buy or make an **IR pen** and an IR-capable camera path. That immediately removes the entire “did the child mean to touch the wall or wave near it?” ambiguity. Refurboard is literally built around that workflow, and Motioncube also documents IR pen control with suitable IR camera hardware. This is the highest fun-per-dollar move if the goal is joy by tomorrow instead of purity by next month. citeturn30view0turn31view0turn34view1

If you are willing to buy **one serious sensor**, pick between a **depth camera** and an **Ultraleap hand tracker** based on what you care about more. A depth camera in the urlAstra 2turn38search1 or urlGemini 2turn37search12 family is the better route if you want real wall-plane logic and eventually true finger-on-wall drawing. Current commercial projection stacks explicitly center depth-camera installs around supported devices from those families. An Ultraleap camera is the better route if you care more about robust **mid-air gestures** and hand tracking quality, especially on Mac via Hyperion, but it does not magically solve large-wall contact semantics on its own. citeturn13view0turn13view1turn34view0turn34view1turn39view0turn39view1turn39view3

One small but important trap: **reset projector keystone and digital zoom before calibration**. TouchDesigner’s camSchnappr docs explicitly warn to make sure digital projector transforms are reset before mapping. If you calibrate on top of projector-side distortions, you are building on quicksand. citeturn36view2

## PRD you can hand to Claude Code

**Product name:** WallSketch

**Product summary:** A kid-friendly projected drawing wall for macOS-first deployment. A projector displays a full-screen canvas on a wall. A tracking service turns an IR pen, fingertip, or tracked hand into drawing input. Simple gestures change tools, colors, and modes. The system must work offline and feel immediate rather than “technically interesting but emotionally dead.”

**Primary user story:** A child walks up to the wall, sees a huge canvas, and can start drawing within seconds. She should not need to navigate menus, hold a headset, or understand modes. The system should feel like a magical wall, not like desktop software that lost a fight.

**Secondary user story:** A parent can re-run calibration in under two minutes, choose between IR pen mode and hand mode, and recover from lighting or camera changes without code changes.

**Goals**
- Immediate visual feedback on a projected wall.
- Large, forgiving interaction targets.
- Reliable drawing even when lighting is imperfect.
- Gesture-based tool changes that are simple and hard to trigger accidentally.
- Clean local architecture: one tracker process, one renderer process, one saved calibration profile.
- macOS-first deployment, but browser/Electron UI should stay cross-platform-friendly.

**Non-goals**
- Perfect multi-user hand segmentation on day one.
- Full-body gesture language.
- Quest-controller-first interaction.
- Fancy projection-mapped arbitrary 3D surfaces in the first release.
- Enterprise classroom software features, cloud sync, or network collaboration.

**Functional requirements**
- Full-screen drawing canvas on the projector display.
- Calibration wizard that stores wall-to-projector mapping and camera intrinsics.
- Input mode switcher:
  - IR pen mode.
  - Hybrid hand mode: hand for gestures, fingertip or stylus for ink.
  - Experimental camera-only hand drawing mode.
- Radial or docked tool palette with oversized targets.
- Brush controls: color, size, soft/hard brush, eraser, clear, undo, save snapshot.
- Hover cursor that appears before contact.
- Stroke smoothing and jitter suppression.
- Optional “magic” brushes for delight: sparkle, rainbow, glow, stamp brush.
- Safe pause or lock mode so a parent can recalibrate or change settings without random wall graffiti.

**Gesture requirements**
- Open palm: show or hide tool palette.
- Pinch: select highlighted tool.
- Thumbs up: undo.
- Fist held for one second: clear confirmation prompt.
- Gestures must be disabled while actively inking, unless explicitly whitelisted.

**Technical requirements**
- Tracker backend in Python.
- Renderer in browser or Electron.
- Local WebSocket channel between tracker and renderer.
- Detector interface abstraction so you can swap between YOLO stylus detection, MediaPipe hand landmarks, and future depth/IR hardware adapters.
- JSON config for calibration, camera choice, brush defaults, and gesture bindings.
- Frame timestamps carried through the pipeline for latency debugging.
- Local-only by default. No cloud dependency.

**Quality targets**
- Subjectively immediate interaction.
- No obvious cursor “teleporting.”
- Calibration survives app restarts.
- One adult can set up the system without touching code.
- Child can recover from mistakes with one giant undo button and a clear-all confirm.

**Milestones**
- **Milestone one:** IR pen mode on your current projector with saved calibration, plus drawing, erase, undo, save.
- **Milestone two:** hand gestures for tool switching using MediaPipe while keeping IR pen as the ink source.
- **Milestone three:** experimental camera-only fingertip drawing with wall-contact gating from a second camera or depth sensor.
- **Milestone four:** polish pass with fun brushes, sound, animations, and session replay.

**Suggested Claude Code work breakdown**
- Build `tracker/calibration.py`.
- Build `tracker/detectors/yolo_tip.py`.
- Build `tracker/gestures/mediapipe_gestures.py`.
- Build `tracker/server.py`.
- Build `renderer/index.html`, `renderer/app.ts`, `renderer/brush.ts`.
- Build `shared/protocol.ts` or JSON schema equivalent.
- Build `settings/calibration_profile.json`.
- Add a fake-input simulator so the renderer can be developed without a camera.

## Starter code stubs

The following stubs match the architecture above: Python owns tracking and calibration; the renderer owns brushes and UI.

```python
# tracker_backend.py
# Minimal Python sidecar for WallSketch.
# Replace detect_tip_with_yolo() and detect_gesture_with_mediapipe()
# with your real implementations.

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Set

import cv2
import numpy as np
import websockets

WS_HOST = "127.0.0.1"
WS_PORT = 8765

# Replace with your saved 3x3 homography after calibration.
H = np.eye(3, dtype=np.float32)


@dataclass
class PointerEvent:
    x: float
    y: float
    contact: bool
    confidence: float
    timestamp_ms: int


def apply_homography(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H @ p
    if abs(float(q[2])) < 1e-6:
        raise ValueError("Homography produced invalid coordinates.")
    q /= q[2]
    return float(q[0]), float(q[1])


def detect_tip_with_yolo(frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Return (x, y, confidence) in camera pixels.
    Stub only. Replace with:
      - your custom YOLO tip/stylus detector, or
      - simple IR blob logic if running pen mode.
    """
    return None


def detect_gesture_with_mediapipe(frame_bgr: np.ndarray) -> Optional[str]:
    """
    Return a gesture command string like:
      'menu_toggle', 'undo', 'clear_arm', 'eraser', etc.
    Stub only. Replace with MediaPipe Gesture Recognizer / Hand Landmarker.
    """
    return None


async def broadcast(clients: Set[websockets.WebSocketServerProtocol], payload: dict) -> None:
    if not clients:
        return
    message = json.dumps(payload)
    dead = []
    for ws in clients:
        try:
            await ws.send(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def main() -> None:
    clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handler(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    server = await websockets.serve(handler, WS_HOST, WS_PORT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0.")

    # Ask politely for lower latency if camera/driver supports it.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            now_ms = int(time.time() * 1000)

            # Gesture channel
            gesture = detect_gesture_with_mediapipe(frame)
            if gesture:
                await broadcast(
                    clients,
                    {
                        "type": "command",
                        "command": gesture,
                        "timestamp_ms": now_ms,
                    },
                )

            # Ink / pointer channel
            det = detect_tip_with_yolo(frame)
            if det is not None:
                cam_x, cam_y, conf = det
                draw_x, draw_y = apply_homography(cam_x, cam_y, H)

                event = PointerEvent(
                    x=draw_x,
                    y=draw_y,
                    contact=True,  # Replace with real contact logic in hand mode.
                    confidence=conf,
                    timestamp_ms=now_ms,
                )

                await broadcast(
                    clients,
                    {
                        "type": "pointer",
                        "x": event.x,
                        "y": event.y,
                        "contact": event.contact,
                        "confidence": event.confidence,
                        "timestamp_ms": event.timestamp_ms,
                    },
                )

            await asyncio.sleep(0)  # yield to event loop
    finally:
        cap.release()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
```

```html
<!-- index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>WallSketch</title>
  <style>
    html, body, canvas {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #111;
      cursor: none;
    }
    #hud {
      position: fixed;
      top: 16px;
      left: 16px;
      color: white;
      font: 16px system-ui, sans-serif;
      background: rgba(0,0,0,0.35);
      padding: 10px 14px;
      border-radius: 12px;
      user-select: none;
    }
  </style>
</head>
<body>
  <canvas id="wall"></canvas>
  <div id="hud">WallSketch</div>

  <script type="module">
    const canvas = document.getElementById("wall");
    const ctx = canvas.getContext("2d");
    const hud = document.getElementById("hud");

    let brush = {
      color: "#ff5ea8",
      size: 18,
      mode: "draw", // draw | erase
    };

    let lastPoint = null;
    let menuVisible = false;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    function drawSegment(a, b) {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = brush.size;

      if (brush.mode === "erase") {
        ctx.globalCompositeOperation = "destination-out";
        ctx.strokeStyle = "rgba(0,0,0,1)";
      } else {
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = brush.color;
      }

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }

    function handleCommand(cmd) {
      switch (cmd) {
        case "menu_toggle":
          menuVisible = !menuVisible;
          hud.textContent = menuVisible ? "Menu open" : "WallSketch";
          break;
        case "undo":
          // Real app: implement a stroke stack and redraw.
          hud.textContent = "Undo";
          break;
        case "eraser":
          brush.mode = brush.mode === "erase" ? "draw" : "erase";
          hud.textContent = brush.mode === "erase" ? "Eraser" : "Brush";
          break;
        case "clear_arm":
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          hud.textContent = "Cleared";
          break;
      }
    }

    resize();
    window.addEventListener("resize", resize);

    const ws = new WebSocket("ws://127.0.0.1:8765");

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "command") {
        handleCommand(msg.command);
        return;
      }

      if (msg.type === "pointer") {
        const point = { x: msg.x, y: msg.y };

        if (msg.contact) {
          if (lastPoint) {
            drawSegment(lastPoint, point);
          }
          lastPoint = point;
        } else {
          lastPoint = null;
        }
      }
    };

    ws.onopen = () => {
      hud.textContent = "Connected";
      setTimeout(() => {
        hud.textContent = "WallSketch";
      }, 1000);
    };

    ws.onerror = () => {
      hud.textContent = "Tracker error";
    };
  </script>
</body>
</html>
```

If I were handing this off in the morning, I would tell Claude Code to build **Milestone one only** first: pen mode, calibration, drawing, erase, undo, save. Once that feels magical, bolt on gesture control. Doing both at once is how prototype schedules become crime scenes.

## Open questions and limitations

A few things remain genuinely dependent on your room and hardware, not on theory. If your current webcams are only available as higher-latency network streams instead of direct camera devices, the custom hand mode will be noticeably worse than the same code on direct USB capture. citeturn30view0

Commercial turnkey stacks are currently stronger on **Windows** for depth-camera and touchless-overlay workflows than on macOS. That does not block a Mac custom build; it just means the Mac path is more “build the exact thing you want” and less “buy a black box and press go.” citeturn34view1turn39view0turn39view1

The biggest unknown for the pure camera-hand version is not model quality. It is **contact logic under front projection**, because front projection introduces shadow and occlusion by design. If you want reliable actual wall-touch behavior with bare hands, a second side camera or a depth sensor is the clean answer. citeturn18view1turn16view1turn16view3

So the clear course of action is this: **ship an IR-pen version immediately, build gestures second, and only then decide whether true finger-on-wall drawing is worth adding with a second camera or depth hardware**. That is the shortest path to “this is magical” instead of “this is an interesting failure mode.” citeturn30view0turn31view0turn14view1turn16view0turn16view1