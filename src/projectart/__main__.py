from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .app import App


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("projectart")
    p.add_argument(
        "--input",
        choices=["mouse", "gloves", "wand", "androidtv"],
        default="mouse",
        help="input source (default: mouse — useful for dev without cameras)",
    )
    p.add_argument("--canvas", default="1920x1080", help="canvas WxH")
    p.add_argument("--ws-host", default="127.0.0.1")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--http-port", type=int, default=8000)
    p.add_argument("--webcam-a", default=None, help="rtsp/url for cam A (10.0.0.33)")
    p.add_argument("--webcam-b", default=None, help="rtsp/url for cam B (10.0.0.34)")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--log", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    w, h = (int(x) for x in args.canvas.split("x"))
    app = App(args=args, canvas_size=(w, h))
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
