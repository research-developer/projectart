"""Calibration persistence — load/save `~/.projectart/calib.json`.

Schema matches `docs/superpowers/specs/2026-05-09-projectart-design.md` §6.4.
We use pydantic for validation so a typo'd or partial file fails loudly
instead of silently mis-aiming the wall.

Numpy arrays are serialized as nested lists (json-friendly). On load we
return numpy arrays for downstream geometry code.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..geometry.stage import StageCalibration

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)

CALIB_VERSION = 1
DEFAULT_CALIB_PATH = Path(
    os.environ.get("PROJECTART_CALIB", "~/.projectart/calib.json")
).expanduser()


def _matrix_validator(rows: int, cols: int):
    """Build a pydantic field validator that asserts a 2D list has the right shape."""

    def check(v):
        if v is None:
            return v
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (rows, cols):
            raise ValueError(f"expected shape ({rows},{cols}); got {arr.shape}")
        return arr.tolist()

    return check


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    K: list[list[float]] = Field(min_length=3, max_length=3)
    dist: list[float] = Field(min_length=4, max_length=14)


class StereoExtrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    R_a_to_b: list[list[float]] = Field(min_length=3, max_length=3)
    t_a_to_b: list[float] = Field(min_length=3, max_length=3)


class WallPlaneSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normal: list[float] = Field(min_length=3, max_length=3)
    centroid: list[float] = Field(min_length=3, max_length=3)


class UvBasisSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    u: list[float] = Field(min_length=3, max_length=3)
    v: list[float] = Field(min_length=3, max_length=3)


class CanvasSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    w: int = 1920
    h: int = 1080


class StageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cam_to_stage: list[list[float]] = Field(min_length=3, max_length=3)
    stage_to_projector: list[list[float]] = Field(min_length=3, max_length=3)
    height_offset: list[float] = Field(
        default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2
    )

    @field_validator("cam_to_stage", "stage_to_projector")
    @classmethod
    def _check_3x3(cls, v):
        return _matrix_validator(3, 3)(v)


class CalibrationDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = CALIB_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    canvas: CanvasSize = Field(default_factory=CanvasSize)
    contact_epsilon_m: float = 0.015

    camera_a: CameraIntrinsics | None = None
    camera_b: CameraIntrinsics | None = None
    stereo: StereoExtrinsics | None = None
    wall_plane: WallPlaneSchema | None = None
    uv_basis: UvBasisSchema | None = None
    stage: StageSchema | None = None

    # Homographies are 3x3 lists of lists. None when not yet calibrated.
    homography_uv_to_canvas: list[list[float]] | None = None
    homography_cam_a_to_canvas: list[list[float]] | None = None  # M2 4-corner shortcut


def save_calibration(doc: CalibrationDoc, path: Path | None = None) -> Path:
    """Atomically write the calibration to disk."""
    target = (path or DEFAULT_CALIB_PATH).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(doc.model_dump_json(indent=2))
    os.replace(tmp, target)
    log.info("saved calibration to %s", target)
    return target


def load_calibration(path: Path | None = None) -> CalibrationDoc | None:
    """Load + validate the calibration, or return None if no file exists."""
    target = (path or DEFAULT_CALIB_PATH).expanduser()
    if not target.exists():
        log.debug("no calibration at %s", target)
        return None
    try:
        raw = json.loads(target.read_text())
        doc = CalibrationDoc.model_validate(raw)
    except Exception:
        log.exception("failed to load calibration at %s; ignoring", target)
        return None
    if doc.version != CALIB_VERSION:
        log.warning(
            "calibration at %s has version=%d but code expects %d; recalibration recommended",
            target,
            doc.version,
            CALIB_VERSION,
        )
    return doc


def empty_for_canvas(canvas_w: int, canvas_h: int) -> CalibrationDoc:
    """Convenience constructor for a fresh CalibrationDoc with just the canvas size set."""
    return CalibrationDoc(canvas=CanvasSize(w=canvas_w, h=canvas_h))


def stage_calibration_from_doc(doc: CalibrationDoc) -> StageCalibration | None:
    """Build a StageCalibration from a doc's stage section, or None if absent."""
    from ..geometry.stage import StageCalibration

    if doc.stage is None:
        return None
    return StageCalibration(
        cam_to_stage=np.asarray(doc.stage.cam_to_stage, dtype=np.float64),
        stage_to_projector=np.asarray(doc.stage.stage_to_projector, dtype=np.float64),
        height_offset=(float(doc.stage.height_offset[0]), float(doc.stage.height_offset[1])),
    )


def doc_with_stage(doc: CalibrationDoc, cal: StageCalibration) -> CalibrationDoc:
    """Return a copy of `doc` with its stage section set from a StageCalibration."""
    stage = StageSchema(
        cam_to_stage=[list(map(float, row)) for row in cal.cam_to_stage],
        stage_to_projector=[list(map(float, row)) for row in cal.stage_to_projector],
        height_offset=[float(cal.height_offset[0]), float(cal.height_offset[1])],
    )
    return doc.model_copy(update={"stage": stage})
