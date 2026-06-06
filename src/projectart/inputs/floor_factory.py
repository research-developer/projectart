"""Build a FloorGameSource from CLI args + persisted calibration."""
from __future__ import annotations

import logging

from ..calibration.persist import load_calibration, stage_calibration_from_doc
from ..games.whack_a_mole import WhackConfig, WhackGame
from ..geometry.stage import StageCalibration
from ..inputs.floor_game import FloorGameSource
from ..server.ws import Server

log = logging.getLogger(__name__)


def build_floor_source(
    canvas_size: tuple[int, int],
    server: Server,
    game: str,
    camera_index: int,
) -> FloorGameSource:
    if game != "whack":
        log.warning("unknown game %r; defaulting to whack", game)
    calib = None
    doc = load_calibration()
    if doc is not None:
        calib = stage_calibration_from_doc(doc)
    if calib is None:
        log.warning(
            "no stage calibration found; using identity (uncalibrated — wrong but visible)"
        )
        calib = StageCalibration.identity(
            frame_w=1280, frame_h=720, proj_w=canvas_size[0], proj_h=canvas_size[1]
        )
    return FloorGameSource(
        server=server,
        calib=calib,
        game=WhackGame(WhackConfig()),
        camera_index=camera_index,
    )
