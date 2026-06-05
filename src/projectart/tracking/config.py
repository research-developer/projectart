from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackingConfig:
    tracker: str = "bytetrack"
    conf: float = 0.25
    iou: float = 0.5
    confirm_after_hits: int = 3
    lost_after_s: float = 0.5
    gone_after_s: float = 2.0
    min_confidence: float = 0.0
    vel_mincutoff: float = 1.0
    vel_beta: float = 0.05

    @classmethod
    def from_dict(cls, d: dict | None) -> TrackingConfig:
        d = d or {}
        vs = d.get("velocity_smoothing", {})
        return cls(
            tracker=d.get("tracker", "bytetrack"),
            conf=float(d.get("conf", 0.25)),
            iou=float(d.get("iou", 0.5)),
            confirm_after_hits=int(d.get("confirm_after_hits", 3)),
            lost_after_s=float(d.get("lost_after_s", 0.5)),
            gone_after_s=float(d.get("gone_after_s", 2.0)),
            min_confidence=float(d.get("min_confidence", 0.0)),
            vel_mincutoff=float(vs.get("mincutoff", 1.0)),
            vel_beta=float(vs.get("beta", 0.05)),
        )
