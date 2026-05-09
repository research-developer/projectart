"""Mouse input source — the dev/CI path that proves the wire contract.

Drawing with the mouse on the canvas page should produce identical PointerEvents
to what the gloves source will produce. Anything that works here will work for
the real input modes.

We listen for mouse events INSIDE the renderer via window.addEventListener and
forward them back over the same websocket as text frames. That way we don't
need pynput / pyobjc / accessibility permissions on macOS for the dev path —
the browser already has the events.
"""
from __future__ import annotations

import asyncio
import json
import logging

from ..server.protocol import HudAnchorEvent, PointerEvent
from ..server.ws import Server

log = logging.getLogger(__name__)


class MouseSource:
    def __init__(self, canvas_size: tuple[int, int], server: Server):
        self.canvas_size = canvas_size
        self.server = server

    async def run(self) -> None:
        # Renderer sends `{type: "mouse", x, y, contact}` upstream over the same WS.
        # We subscribe by monkey-patching the server's _handle_client to also
        # process inbound messages.
        original = self.server._handle_client

        async def handle(ws):
            self.server._clients.add(ws)
            try:
                # Send the canvas hello first (mirroring Server._handle_client).
                from ..server.protocol import Hello
                from ..server.ws import _serialize
                hello = Hello(canvas_w=self.canvas_size[0], canvas_h=self.canvas_size[1])
                await ws.send(_serialize(hello))

                last_x = last_y = 0.0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("type") != "mouse":
                        continue
                    x = float(msg.get("x", 0))
                    y = float(msg.get("y", 0))
                    contact = bool(msg.get("contact", False))
                    ts_ms = int(msg.get("ts_ms", 0))
                    velocity = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                    last_x, last_y = x, y

                    pe = PointerEvent(
                        x=x, y=y, contact=contact, velocity=velocity, ts_ms=ts_ms
                    )
                    he = HudAnchorEvent(x=x, y=y, visible=True, ts_ms=ts_ms)
                    await self.server.broadcast(pe)
                    await self.server.broadcast(he)
            finally:
                self.server._clients.discard(ws)

        # Replace the server's handler. Since websockets.serve was already
        # bound to the original handler, we need to restart it.
        self.server.close()
        await asyncio.sleep(0.05)

        import websockets
        self.server._ws_server = await websockets.serve(
            handle, self.server.ws_host, self.server.ws_port
        )
        # Re-start http (close() shut it down too)
        self.server._start_http()
        log.info("mouse input source running. Draw on the canvas with left-button held.")

        await self.server._ws_server.wait_closed()
