"""YOLO dot/hand detector.

Wraps `ultralytics.YOLO` with a small dataclass output. The custom weights
trained for our dot gloves (claimed 6× lower memory, ~40% faster than
baseline) have not yet been located on this machine — see Linear IMA-210.
Until they're confirmed, this module accepts an explicit path or falls
back to `yolov8n.pt` so M2/M3 development isn't blocked.

Usage:
    detector = DotDetector(weights_path=Path("models/yolo-dots.pt"))
    detections = detector(frame_bgr)   # list[Detection]
    # or with persistent tracking:
    detections = detector.track(frame_bgr)  # list[Detection] with track_id
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


def _parse_boxes(boxes, class_names: dict[int, str], with_ids: bool) -> list[Detection]:
    """Pure converter from an ultralytics Boxes-like object to Detections.
    Accepts anything exposing .xywh/.cls/.conf/.id with .cpu().numpy()."""
    if boxes is None or len(boxes) == 0:
        return []

    def arr(t):
        return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)

    xywh = arr(boxes.xywh)
    cls = arr(boxes.cls).astype(int)
    conf = arr(boxes.conf)
    ids = None
    if with_ids and getattr(boxes, "id", None) is not None:
        ids = arr(boxes.id).astype(int)
    out: list[Detection] = []
    for i in range(len(xywh)):
        cx, cy, w, h = (float(v) for v in xywh[i])
        cid = int(cls[i])
        out.append(
            Detection(
                class_id=cid, cx=cx, cy=cy, w=w, h=h,
                confidence=float(conf[i]),
                class_name=class_names.get(cid, ""),
                track_id=(int(ids[i]) if ids is not None else None),
            )
        )
    return out


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
            from ultralytics import YOLO
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
        results = self._model(frame_bgr, imgsz=self.imgsz, conf=self.conf_thresh, verbose=False)
        if not results:
            return []
        return _parse_boxes(results[0].boxes, self._class_names, with_ids=False)

    _TRACKER_YAML = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}

    def track(self, frame_bgr: np.ndarray, tracker: str = "bytetrack") -> list[Detection]:
        """Run detection + persistent tracking. Each Detection carries a stable
        `track_id` across calls (ByteTrack/BoT-SORT)."""
        self._ensure_loaded()
        yaml = self._TRACKER_YAML.get(tracker, "bytetrack.yaml")
        results = self._model.track(
            frame_bgr, persist=True, tracker=yaml,
            imgsz=self.imgsz, conf=self.conf_thresh, verbose=False,
        )
        if not results:
            return []
        return _parse_boxes(results[0].boxes, self._class_names, with_ids=True)
