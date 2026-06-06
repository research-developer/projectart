from projectart.games.whack_a_mole import WhackConfig, WhackGame
from projectart.geometry.stage import StageCalibration
from projectart.inputs.floor_game import FloorGameSource, coast_markers


def _src():
    cal = StageCalibration.identity(frame_w=640, frame_h=360, proj_w=1920, proj_h=1080)
    game = WhackGame(WhackConfig(spawn_interval_s=100.0, seed=1))
    return FloorGameSource.for_testing(cal, game)


def test_step_emits_scene_and_gamestate_with_mole_in_stage_coords():
    src = _src()
    scene, gs = src.step([], ts=0.0)          # spawns one mole at t=0
    assert gs.type == "game_state"
    moles = [o for o in scene.objects if o.kind == "mole"]
    assert len(moles) == 1
    assert 0.0 <= moles[0].x <= 1.0 and 0.0 <= moles[0].y <= 1.0


def test_step_maps_marker_camera_px_to_stage_and_scores():
    src = _src()
    src.step([], ts=0.0)
    mole = next(iter(src.game.moles.values()))
    # camera px for that stage point under identity 640x360:
    cx, cy = mole.x * 640, mole.y * 360
    scene, gs = src.step([(1, cx, cy)], ts=0.1)
    assert gs.score == 1


def test_coast_markers_extrapolates_then_drops():
    state = {}
    # frame 1: marker seen at (0.2,0.2)
    out = coast_markers(state, [(1, 0.2, 0.2)], ts=0.0, max_coast_s=0.2)
    assert out == [(1, 0.2, 0.2)]
    # frame 2: seen at (0.4,0.2) -> velocity +0.2/s in x... dt 0.1 -> vx=2.0
    out = coast_markers(state, [(1, 0.4, 0.2)], ts=0.1, max_coast_s=0.2)
    assert out == [(1, 0.4, 0.2)]
    # frame 3: DROPOUT -> coast forward by vx*dt = 2.0*0.1 = 0.2 -> x~0.6
    out = coast_markers(state, [], ts=0.2, max_coast_s=0.2)
    assert len(out) == 1 and abs(out[0][1] - 0.6) < 1e-6
    # long gap beyond max_coast_s -> dropped
    out = coast_markers(state, [], ts=1.0, max_coast_s=0.2)
    assert out == []
