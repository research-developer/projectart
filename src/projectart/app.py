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
        log.info(
            "ProjectArt %s — input=%s canvas=%dx%d", "0.1.0", self.args.input, *self.canvas_size
        )

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
        elif self.args.input == "gloves":
            from .inputs.gloves import build_gloves_source

            source = build_gloves_source(
                canvas_size=self.canvas_size,
                server=self._server,
                webcam_a=self.args.webcam_a,
                webcam_b=self.args.webcam_b,
                yolo_weights=self.args.yolo_weights,
            )
            await source.run()
        elif self.args.input == "scene":
            from .inputs.scene import build_scene_source

            source = build_scene_source(
                canvas_size=self.canvas_size,
                server=self._server,
                webcam_a=self.args.webcam_a,
                webcam_b=self.args.webcam_b,
                yolo_weights=self.args.yolo_weights,
                audio_device=self.args.audio_device,
                enable_freeze_game=getattr(self.args, "freeze", False),
                camera_index=self.args.camera_index,
            )
            await source.run()
        elif self.args.input == "reactive":
            from .inputs.reactive import build_reactive_source

            source = build_reactive_source(
                canvas_size=self.canvas_size,
                server=self._server,
                webcam_a=self.args.webcam_a,
                yolo_weights=self.args.yolo_weights,
                config_path=getattr(self.args, "reactive_config", None),
            )
            await source.run()
        elif self.args.input == "floor":
            from .inputs.floor_factory import build_floor_source

            source = build_floor_source(
                canvas_size=self.canvas_size,
                server=self._server,
                game=self.args.game,
                camera_index=self.args.camera_index if self.args.camera_index is not None else 0,
            )
            await source.run()
        else:
            log.warning(
                "input=%s not yet implemented; idling.", self.args.input
            )
            await self._server.wait_closed()
