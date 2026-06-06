"""Floor-game input source: local camera -> ArUco -> stage -> WhackGame ->
SceneFrame + GameState. `step()` is pure (no camera) for tests; `run()` adds the
live loop with marker coasting through brief detection dropouts.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..calibration.persist import (
    CalibrationDoc,
    doc_with_stage,
    load_calibration,
    save_calibration,
)
from ..games.whack_a_mole import WhackGame
from ..geometry.stage import STAGE_CORNERS, StageCalibration, _homography
from ..server.protocol import GameState, SceneFrame, SceneObject
from ..server.ws import Server

log = logging.getLogger(__name__)


def coast_markers(
    state: dict,
    detected: list[tuple[int, float, float]],
    ts: float,
    max_coast_s: float = 0.15,
) -> list[tuple[int, float, float]]:
    """Carry markers through short dropouts. `state` is caller-owned and holds
    per-id (x, y, vx, vy, ts, seen_ts). Detected markers update velocity; missing
    ones are extrapolated until `max_coast_s` since last real detection, then dropped.
    Returns list of (id, x, y) in the same coordinate space as `detected`."""
    out: list[tuple[int, float, float]] = []
    seen_ids: set[int] = set()
    for mid, x, y in detected:
        seen_ids.add(mid)
        prev = state.get(mid)
        vx = vy = 0.0
        if prev is not None:
            dt = max(1e-3, ts - prev["ts"])
            vx, vy = (x - prev["x"]) / dt, (y - prev["y"]) / dt
        state[mid] = {"x": x, "y": y, "vx": vx, "vy": vy, "ts": ts, "seen_ts": ts}
        out.append((mid, x, y))
    for mid, s in list(state.items()):
        if mid in seen_ids:
            continue
        if ts - s["seen_ts"] > max_coast_s:
            del state[mid]
            continue
        dt = ts - s["ts"]
        nx, ny = s["x"] + s["vx"] * dt, s["y"] + s["vy"] * dt
        s["x"], s["y"], s["ts"] = nx, ny, ts
        out.append((mid, nx, ny))
    return out


class FloorGameSource:
    def __init__(
        self,
        server: Server | None,
        calib: StageCalibration,
        game: WhackGame,
        camera_index: int = 0,
        frame_size: tuple[int, int] = (1280, 720),
        target_hz: int = 60,
    ):
        self.server = server
        self.calib = calib
        self.game = game
        self.camera_index = camera_index
        self.frame_w, self.frame_h = frame_size
        self.target_hz = target_hz
        self._period = 1.0 / max(1, target_hz)
        self._coast: dict = {}
        self._capture = None
        self._detector = None
        self._pending_input_corner: int | None = None
        self._input_cam_corners: list[tuple[float, float] | None] = [None, None, None, None]

    @classmethod
    def for_testing(cls, calib: StageCalibration, game: WhackGame) -> FloorGameSource:
        return cls(server=None, calib=calib, game=game)

    # ---- calibration (driven by the renderer over the WS inbound channel) ----

    def _persist_calib(self) -> None:
        doc = load_calibration() or CalibrationDoc()
        save_calibration(doc_with_stage(doc, self.calib))

    def _on_calib_message(self, msg: dict) -> None:
        """Handle calibration commands from the renderer."""
        mtype = msg.get("type")
        if mtype == "calib_output":
            # 4 projector px (stage order TL,TR,BR,BL) the user dragged the stage to.
            try:
                self.calib.stage_to_projector = _homography(STAGE_CORNERS, msg["corners"])
                self._persist_calib()
                log.info("output calibration updated + saved")
            except Exception:
                log.exception("bad calib_output message")
        elif mtype == "calib_input_capture":
            self._pending_input_corner = int(msg.get("corner", 0))
            log.info("will capture next marker for stage corner %d", self._pending_input_corner)

    def _maybe_capture_input_corner(self, detected: list[tuple[int, float, float]]) -> None:
        """If the renderer requested an input-corner capture, record the next marker's
        camera px for that corner; once all 4 are set, rebuild camera->stage (the marker
        is placed at PLAYING HEIGHT on each projected corner — see the spec)."""
        if self._pending_input_corner is None or not detected:
            return
        i = self._pending_input_corner
        self._input_cam_corners[i] = (detected[0][1], detected[0][2])
        self._pending_input_corner = None
        log.info("captured camera corner %d at %s", i, self._input_cam_corners[i])
        if all(c is not None for c in self._input_cam_corners):
            proj_corners = [self.calib.stage_to_proj_px(sx, sy) for sx, sy in STAGE_CORNERS]
            self.calib = StageCalibration.from_corners(
                self._input_cam_corners, STAGE_CORNERS, proj_corners,
                height_offset=self.calib.height_offset,
            )
            self._persist_calib()
            log.info("input calibration (camera->stage) updated + saved")

    def step(
        self, markers_px: list[tuple[int, float, float]], ts: float
    ) -> tuple[SceneFrame, GameState]:
        stage_markers = [
            (mid, *self.calib.cam_px_to_stage(cx, cy)) for mid, cx, cy in markers_px
        ]
        moles = self.game.tick(stage_markers, ts)
        objs = [
            SceneObject(
                id=mo.id,
                kind="mole",
                shape="circle",
                x=mo.x,
                y=mo.y,
                r=self.game.cfg.hit_radius,
                color="#cc3333",
                alpha=1.0,
            )
            for mo in moles
        ]
        ts_ms = int(ts * 1000)
        scene = SceneFrame(ts_ms=ts_ms, objects=objs)
        gs = GameState(
            score=self.game.score,
            round_ms_left=self.game.time_left_ms(ts),
            phase=self.game.phase,
            ts_ms=ts_ms,
        )
        return scene, gs

    async def run(self) -> None:
        from ..capture.local_cam import LocalCamera
        from ..detection.aruco import ArucoDetector

        assert self.server is not None
        self._capture = LocalCamera(index=self.camera_index, name="local")
        self._detector = ArucoDetector()

        async def _ws_handler(_ws, msg):
            self._on_calib_message(msg)

        self.server.set_message_handler(_ws_handler)
        self._capture.start()
        log.info(
            "floor game starting (camera_index=%d, target_hz=%d)",
            self.camera_index,
            self.target_hz,
        )
        loop_t0 = time.monotonic()
        try:
            while True:
                t0 = time.monotonic()
                frame = self._capture.latest()
                if frame is None:
                    await asyncio.sleep(self._period)
                    continue
                self.frame_h, self.frame_w = frame.image.shape[:2]
                ts = time.monotonic() - loop_t0
                detected = [(m.id, m.cx, m.cy) for m in self._detector(frame.image)]
                self._maybe_capture_input_corner(detected)
                # coast in CAMERA px then map to stage in step()
                coasted = coast_markers(self._coast, detected, ts)
                scene, gs = self.step(coasted, ts)
                await self.server.broadcast(scene)
                await self.server.broadcast(gs)
                await asyncio.sleep(max(0.0, self._period - (time.monotonic() - t0)))
        finally:
            self._capture.stop()
