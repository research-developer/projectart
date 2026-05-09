# projectart — Claude Code instructions

- Language: Python 3.11. Single venv. Use `pyproject.toml`, no requirements.txt.
- Style: `ruff` (configured in pyproject); 100-col max. Type-hint public functions.
- Logging via `logging.getLogger(__name__)` — no `print()` in library code.
- Filesystem paths go through `projectart.config.paths` (helper to be added in M5).
- Hot paths use numpy ops, not Python loops (filter, raycast, stereo, stroke buffer).
- Input sources stay swappable. Never import `cv2`, `ultralytics`, or `openvr` outside their dedicated modules:
    - `cv2` — capture/, detection/, calibration/, geometry/
    - `ultralytics` — detection/yolo_dots.py only
    - `pynput` — inputs/mouse.py only (and only if used; renderer-side mouse is preferred)
- WebSocket protocol lives in `server/protocol.py`. Any change to the wire format bumps `PROTOCOL_VERSION` and requires renderer-side handling.
- Renderer code is plain JS in `renderer/`. p5.js + p5.brush.js (CDN in MVP, vendor later).
- Tests in `tests/` use `pytest`. geom/filter/protocol/stereo MUST be testable without a display or camera.
- Calibration JSON at `~/.projectart/calib.json`. Schema in `docs/superpowers/specs/2026-05-09-projectart-design.md` §6.4.
- Cameras: 10.0.0.33 (cam A) and 10.0.0.34 (cam B). yi-hack-v5. Prefer the low-res RTSP stream for inference.
- When in doubt, prefer the simpler version that works tonight. Optimization waits for a profiling pass.
