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

## Floor games (whack-a-mole)

Project a game onto the floor and play with an ArUco-marked mallet/foot tag.

```bash
python -m projectart --input floor --game whack \
    --camera-index 0 --http-port 8011 --ws-port 8766
# open on the projector display:
#   http://127.0.0.1:8011/floor.html?ws=ws://127.0.0.1:8766/
```

**First-time setup**

- **Camera permission**: grant your terminal Camera access (System Settings → Privacy &
  Security → Camera). The capture thread can't raise the prompt itself.
- **Continuity Camera**: turn **OFF** Center Stage / video effects — per-frame reframing
  breaks the camera→stage calibration. Use a short shutter + good light.
- **Markers**: print an ArUco tag from the `DICT_4X4_50` dictionary for the mallet/foot.

**Calibration** (in the browser, projected on the floor)

- Press **`c`** and drag the 4 green corners until the projected stage is a clean
  rectangle on the floor (saved automatically).
- Press **`1`–`4`** while holding the marker on a block at playing height on each
  projected corner (TL, TR, BR, BL) to set camera→stage. Persists to
  `~/.projectart/calib.json`.

## Status

- M0 ✅ scaffold + ws/http server + black p5.js canvas
- M1 🛠️  mouse input draws strokes via the WS contract (in progress)
- M2 ⏳ single-cam Yi + YOLO dot detection + 4-corner draw
- M3 ⏳ stereo + wall plane + contact gating
- M4 ⏳ floating HUD that follows the active hand
- M5 ⏳ ChArUco calibration + persistence
- M6 ⏳ brushes, sounds, save, undo, clear

Linear project: ProjectArt — interactive wall drawing (Imajn / IMA-203 … IMA-213)
