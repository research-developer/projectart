#!/usr/bin/env python3
"""Smoke-test that ultralytics + yolov8n.pt detects people and cats.

Usage:
    python training/scripts/verify_yolov8.py /path/to/image.jpg
    python training/scripts/verify_yolov8.py /path/to/image.jpg --weights /path/to/best.pt

Prints one line per detection: `class_name conf cx cy w h`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="path to a JPEG/PNG to test")
    p.add_argument("--weights", default=None, help="weights file (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args(argv)

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install -e '.[yolo]'", file=sys.stderr)
        return 2

    model = YOLO(args.weights or "yolov8n.pt")
    results = model(str(args.image), conf=args.conf, imgsz=args.imgsz, verbose=False)

    if not results:
        print("no results")
        return 0

    r = results[0]
    names = getattr(model, "names", {}) or {}
    if r.boxes is None or len(r.boxes) == 0:
        print("(no detections)")
        return 0

    boxes = r.boxes
    xywh = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, "cpu") else boxes.xywh
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else boxes.cls.astype(int)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf

    for i in range(len(xywh)):
        cx, cy, w, h = (float(v) for v in xywh[i])
        c = int(cls[i])
        name = names.get(c, str(c))
        print(f"{name:<12} {float(conf[i]):.3f}  cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
