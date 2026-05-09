from __future__ import annotations

import argparse
import logging
import webbrowser
from pathlib import Path

from .server.ws import Server

log = logging.getLogger(__name__)


class App:
    """Top-level orchestrator. Owns the websocket server and the chosen input source."""

    def __init__(self, args: argparse.Namespace, canvas_size: tuple[int, int]):
        self.args = args
        self.canvas_size = canvas_size
        self._renderer_dir = Path(__file__).resolve().parents[2] / "renderer"
        self._server = Server(
            ws_host=args.ws_host,
            ws_port=args.ws_port,
            http_port=args.http_port,
            static_dir=self._renderer_dir,
            canvas_size=canvas_size,
        )

    async def run(self) -> None:
        log.info("ProjectArt %s — input=%s canvas=%dx%d", "0.1.0", self.args.input, *self.canvas_size)

        await self._server.start()
        log.info(
            "renderer at http://%s:%d/  (ws://%s:%d/)",
            self.args.ws_host,
            self.args.http_port,
            self.args.ws_host,
            self.args.ws_port,
        )
        url = f"http://{self.args.ws_host}:{self.args.http_port}/"
        try:
            webbrowser.open(url)
        except Exception:
            log.warning("could not auto-open browser; visit %s manually", url)

        if self.args.input == "mouse":
            from .inputs.mouse import MouseSource

            source = MouseSource(canvas_size=self.canvas_size, server=self._server)
            await source.run()
        else:
            log.warning("input=%s not yet implemented; idling. (M0 only ships --input mouse)",
                       self.args.input)
            await self._server.wait_closed()
