"""Mouse input source — the dev/CI path that proves the wire contract.

The browser sketch already sees mouse events. Rather than reading the macOS
event stream from Python (which would need accessibility permissions and
pynput / pyobjc), we let the browser forward `{type:"mouse", x, y, contact}`
upstream over the websocket. We translate those into PointerEvents and
broadcast them back as the canonical contract — the same flow gloves/wand
will use later, just with different upstream producers.
"""
from __future__ import annotations

import logging

from ..server.protocol import HudAnchorEvent, PointerEvent
from ..server.ws import Server

log = logging.getLogger(__name__)


class MouseSource:
    def __init__(self, canvas_size: tuple[int, int], server: Server):
        self.canvas_size = canvas_size
        self.server = server
        self._last_xy: tuple[float, float] | None = None

    async def run(self) -> None:
        self.server.set_message_handler(self._on_message)
        log.info("mouse input running. Drag with left button on the canvas to draw.")
        await self.server.wait_closed()

    async def _on_message(self, _ws, msg: dict) -> None:
        if msg.get("type") != "mouse":
            return
        try:
            x = float(msg["x"])
            y = float(msg["y"])
        except (KeyError, TypeError, ValueError):
            return
        contact = bool(msg.get("contact", False))
        ts_ms = int(msg.get("ts_ms", 0))

        velocity = 0.0
        if self._last_xy is not None:
            dx = x - self._last_xy[0]
            dy = y - self._last_xy[1]
            velocity = (dx * dx + dy * dy) ** 0.5
        self._last_xy = (x, y)

        await self.server.broadcast(
            PointerEvent(x=x, y=y, contact=contact, velocity=velocity, ts_ms=ts_ms)
        )
        await self.server.broadcast(
            HudAnchorEvent(x=x, y=y, visible=True, ts_ms=ts_ms)
        )
