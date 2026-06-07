#!/usr/bin/env python3
"""Dev helper: loopback TCP relay to a LAN host via Apple's /usr/bin/nc.

Works around macOS Local-Network blocking of third-party binaries (python/
opencv/ffmpeg) — see memory/lan-blocked-for-thirdparty-binaries. Run this, then
point the app at rtsp://127.0.0.1:8554/<path>.

    python tools/lan_relay.py --listen 127.0.0.1:8554 --to 10.0.0.33:554

Once Local Network permission is granted to your terminal, skip this and point
--webcam-a directly at rtsp://10.0.0.33/ch0_1.h264.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import threading


def _handle(cli: socket.socket, host: str, port: int) -> None:
    nc = subprocess.Popen(["/usr/bin/nc", host, str(port)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def pump(src_read, dst_write, closer):
        try:
            while True:
                d = src_read(65536)
                if not d:
                    break
                dst_write(d)
        except Exception:
            pass
        finally:
            closer()

    threading.Thread(
        target=pump,
        args=(cli.recv, lambda d: (nc.stdin.write(d), nc.stdin.flush()), nc.stdin.close),
        daemon=True,
    ).start()
    threading.Thread(
        target=pump,
        args=(nc.stdout.read1, cli.sendall, lambda: (cli.close(), nc.kill())),
        daemon=True,
    ).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8554")
    ap.add_argument("--to", default="10.0.0.33:554")
    args = ap.parse_args()
    lh, lp = args.listen.split(":")
    th, tp = args.to.split(":")
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lh, int(lp)))
    srv.listen(8)
    print(f"relay {args.listen} -> {args.to} (nc outbound)", flush=True)
    while True:
        cli, _ = srv.accept()
        _handle(cli, th, int(tp))


if __name__ == "__main__":
    main()
