"""YOLO dot/hand detector.

Wraps `ultralytics.YOLO` with a small dataclass output. The custom weights
trained for our dot gloves (claimed 6× lower memory, ~40% faster than
baseline) have not yet been located on this machine — see Linear IMA-210.
Until they're confirmed, this module accepts an explicit path or falls
back to `yolov8n.pt` so M2/M3 development isn't blocked.

Usage:
    detector = DotDetector(weights_path=Path("models/yolo-dots.pt"))
    detections = detector(frame_bgr)   # list[Detection]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FALLBACK = "yolov8n.pt"   # downloaded by ultralytics on first use


@dataclass(slots=True)
class Detection:
    class_id: int
    cx: float          # center x in pixels
    cy: float          # center y in pixels
    w: float           # bbox width in pixels
    h: float           # bbox height in pixels
    confidence: float
    class_name: str = ""
    track_id: int | None = None


class DotDetector:
    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str = "auto",
        imgsz: int = 640,
        conf_thresh: float = 0.25,
    ):
        self.weights_path = (
            Path(weights_path) if weights_path is not None else None
        )
        self.device = device
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self._model = None
        self._class_names: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. `pip install -e '.[yolo]'` "
                "or `pip install ultralytics`."
            ) from e

        path: str
        if self.weights_path and self.weights_path.exists():
            path = str(self.weights_path)
            log.info("loading YOLO weights from %s", path)
        else:
            if self.weights_path is not None:
                log.warning(
                    "weights file %s not found; falling back to %s",
                    self.weights_path,
                    DEFAULT_FALLBACK,
                )
            else:
                log.info(
                    "no --yolo-weights set; falling back to %s (download on first use)",
                    DEFAULT_FALLBACK,
                )
            path = DEFAULT_FALLBACK

        self._model = YOLO(path)
        names = getattr(self._model, "names", None) or {}
        self._class_names = (
            {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {}
        )

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        self._ensure_loaded()
        results = self._model(  # type: ignore[misc]
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes
        xywh = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, "cpu") else np.asarray(boxes.xywh)
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls).astype(int)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        out: list[Detection] = []
        for i in range(len(xywh)):
            cx, cy, w, h = (float(v) for v in xywh[i])
            class_id = int(cls[i])
            out.append(
                Detection(
                    class_id=class_id,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    confidence=float(conf[i]),
                    class_name=self._class_names.get(class_id, ""),
                )
            )
        return out
