"""Floor-game input source: local camera -> ArUco -> stage -> WhackGame ->
SceneFrame + GameState. `step()` is pure (no camera) for tests; `run()` adds the
live loop with marker coasting through brief detection dropouts.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..games.whack_a_mole import WhackGame
from ..geometry.stage import StageCalibration
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

    @classmethod
    def for_testing(cls, calib: StageCalibration, game: WhackGame) -> FloorGameSource:
        return cls(server=None, calib=calib, game=game)

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
                # coast in CAMERA px then map to stage in step()
                coasted = coast_markers(self._coast, detected, ts)
                scene, gs = self.step(coasted, ts)
                await self.server.broadcast(scene)
                await self.server.broadcast(gs)
                await asyncio.sleep(max(0.0, self._period - (time.monotonic() - t0)))
        finally:
            self._capture.stop()
