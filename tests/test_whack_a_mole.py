from projectart.games.whack_a_mole import WhackConfig, WhackGame, seg_point_dist


def _cfg(**kw):
    base = dict(rows=3, cols=3, spawn_interval_s=1.0, mole_lifetime_s=10.0,
                points=1, round_seconds=100.0, hit_radius=0.08, seed=1, margin=0.1)
    base.update(kw)
    return WhackConfig(**base)


def test_spawn_cadence():
    g = WhackGame(_cfg())
    g.tick([], ts=0.0)          # spawns at t=0
    assert len(g.moles) == 1
    g.tick([], ts=0.5)          # before next interval
    assert len(g.moles) == 1
    g.tick([], ts=1.0)          # second spawn
    assert len(g.moles) == 2


def test_mole_expires():
    g = WhackGame(_cfg(mole_lifetime_s=1.0, spawn_interval_s=100.0))
    g.tick([], ts=0.0)
    assert len(g.moles) == 1
    g.tick([], ts=1.0)          # lifetime elapsed -> expired
    assert len(g.moles) == 0


def test_hit_scores_once_and_removes_mole():
    g = WhackGame(_cfg(spawn_interval_s=100.0))
    g.tick([], ts=0.0)
    mole = next(iter(g.moles.values()))
    # marker exactly on the mole
    g.tick([(1, mole.x, mole.y)], ts=0.1)
    assert g.score == 1
    assert mole.id not in g.moles
    # same marker next frame: no double count (mole gone)
    g.tick([(1, mole.x, mole.y)], ts=0.2)
    assert g.score == 1


def test_swept_segment_catches_fast_marker_that_jumps_over_mole():
    g = WhackGame(_cfg(spawn_interval_s=100.0, hit_radius=0.05))
    g.tick([], ts=0.0)
    mole = next(iter(g.moles.values()))
    # Two frames: marker is far on the left, then far on the right — both
    # single points miss, but the segment passes through the mole.
    g.tick([(1, mole.x - 0.3, mole.y)], ts=0.1)
    assert g.score == 0
    g.tick([(1, mole.x + 0.3, mole.y)], ts=0.2)
    assert g.score == 1


def test_miss_does_not_score():
    g = WhackGame(_cfg(spawn_interval_s=100.0, hit_radius=0.02))
    g.tick([], ts=0.0)
    mole = next(iter(g.moles.values()))
    g.tick([(1, mole.x + 0.5, mole.y + 0.5)], ts=0.1)
    assert g.score == 0


def test_round_over():
    g = WhackGame(_cfg(round_seconds=2.0))
    g.tick([], ts=0.0)
    assert g.phase == "playing"
    g.tick([], ts=2.0)
    assert g.phase == "over"


def test_seg_point_dist_endpoint_and_middle():
    assert seg_point_dist(0, 0, 10, 0, 5, 0) == 0.0
    assert seg_point_dist(0, 0, 10, 0, 5, 3) == 3.0
    assert seg_point_dist(0, 0, 0, 0, 3, 4) == 5.0  # degenerate segment = point
