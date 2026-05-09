"""asyncio websocket server + tiny static-file HTTP server for the renderer.

The static handler is intentionally minimal — Python's stdlib does not have a
clean async path for serving files alongside a websockets app, so we run an
http.server in a thread. Renderer is a couple hundred KB; this is fine.
"""
from __future__ import annotations

import asyncio
import http.server
import json
import logging
import socketserver
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path

import websockets
from websockets.server import WebSocketServerProtocol

from .protocol import Hello, PROTOCOL_VERSION

log = logging.getLogger(__name__)


def _serialize(event) -> str:
    if is_dataclass(event):
        return json.dumps(asdict(event))
    if isinstance(event, dict):
        return json.dumps(event)
    raise TypeError(f"cannot serialize {type(event).__name__}")


class Server:
    def __init__(
        self,
        ws_host: str,
        ws_port: int,
        http_port: int,
        static_dir: Path,
        canvas_size: tuple[int, int],
    ):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.http_port = http_port
        self.static_dir = static_dir
        self.canvas_size = canvas_size
        self._clients: set[WebSocketServerProtocol] = set()
        self._ws_server: websockets.server.Serve | None = None
        self._http_server: socketserver.ThreadingTCPServer | None = None
        self._http_thread: threading.Thread | None = None

    async def start(self) -> None:
        self._start_http()
        self._ws_server = await websockets.serve(
            self._handle_client, self.ws_host, self.ws_port
        )
        log.info("websocket listening on ws://%s:%d", self.ws_host, self.ws_port)

    def _start_http(self) -> None:
        static_dir = self.static_dir

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(static_dir), **kwargs)

            def log_message(self, fmt, *args):
                # quiet
                pass

        srv = socketserver.ThreadingTCPServer((self.ws_host, self.http_port), Handler)
        srv.daemon_threads = True
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self._http_server = srv
        self._http_thread = thread
        log.info("http static at http://%s:%d/  (root=%s)", self.ws_host, self.http_port, static_dir)

    async def _handle_client(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        log.info("client connected (%d total)", len(self._clients))
        try:
            hello = Hello(canvas_w=self.canvas_size[0], canvas_h=self.canvas_size[1])
            await ws.send(_serialize(hello))
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)
            log.info("client disconnected (%d total)", len(self._clients))

    async def broadcast(self, event) -> None:
        if not self._clients:
            return
        payload = _serialize(event)
        dead = []
        for ws in self._clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def wait_closed(self) -> None:
        if self._ws_server is not None:
            await self._ws_server.wait_closed()

    def close(self) -> None:
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._ws_server is not None:
            self._ws_server.close()
