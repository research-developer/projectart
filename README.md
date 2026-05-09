# ProjectArt

Interactive wall drawing system for kids.

Two ceiling cameras + dot-marker gloves + custom YOLO + projector. A small floating HUD follows the active hand for tools/colors/brushes.

See `PRD.md` (or `docs/superpowers/specs/2026-05-09-projectart-design.md`) for the full design.

## Quick start

```bash
pip install -e .
python -m projectart --input mouse
```

A black canvas opens in the default browser. Drag with the left mouse button to draw — this is the M1 "wire-contract test" path that the camera modes will reuse.

## CLI

```
python -m projectart [--input mouse|gloves|scene|wand|androidtv]
                     [--canvas WxH] [--ws-host HOST] [--ws-port N] [--http-port N]
                     [--webcam-a URL] [--webcam-b URL]
                     [--yolo-weights PATH] [--recalibrate]
                     [--log LEVEL]
```

### Input modes

| `--input` | What it does |
|---|---|
| `mouse` (default) | Dev/CI path. Drag with the mouse, draw lines. No camera, no YOLO. |
| `gloves` | Dot-glove drawing. M2 single-cam path; M3 stereo activates if calibration loaded. |
| `scene` | Watch the Yi cameras with YOLO; track cats/people/anything in COCO; broadcast EntityEvents so the renderer overlays a labeled box on each tracked entity. |
| `wand` *(planned)* | Single-cam ArUco wand fallback. |
| `androidtv` *(optional)* | ADB control sidecar — only if the projector source is an Android TV. |

## Layout

```
src/projectart/        Python backend (capture, detection, geometry, server, inputs)
renderer/              Browser renderer (p5.js, WebSocket client)
docs/superpowers/specs/    Design spec(s)
research-notes/        Original three research PRDs (synthesized in PRD.md)
```

## Status

- M0 ✅ scaffold + ws/http server + black p5.js canvas
- M1 🛠️  mouse input draws strokes via the WS contract (in progress)
- M2 ⏳ single-cam Yi + YOLO dot detection + 4-corner draw
- M3 ⏳ stereo + wall plane + contact gating
- M4 ⏳ floating HUD that follows the active hand
- M5 ⏳ ChArUco calibration + persistence
- M6 ⏳ brushes, sounds, save, undo, clear

Linear project: ProjectArt — interactive wall drawing (Imajn / IMA-203 … IMA-213)
