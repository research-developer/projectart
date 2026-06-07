"""Tests for the FreezeGame trigger (registry-driven, no camera/display/audio).

All timestamps are small explicit floats; config values keep rounds short and
deterministic.  We use explicit ``track_id`` on every Detection so the registry's
track-id association path is exercised and entities are never accidentally merged.
"""
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.games.freeze_tag import FreezeConfig, FreezeGame
from projectart.tracking import TrackedRegistry
from projectart.tracking.builtins import Cat, Person

# A small config that makes rounds fast in tests.
FAST = FreezeConfig(
    settle_s=0.2,
    freeze_window_s=0.5,
    round_cooldown_s=0.5,
    move_tolerance=0.1,
)


def _reg() -> TrackedRegistry:
    return TrackedRegistry(entity_types=[Cat, Person])


def _person(track_id: int, cx: float, cy: float, w: float = 60.0, h: float = 120.0) -> Detection:
    return Detection(
        class_id=0, cx=cx, cy=cy, w=w, h=h,
        confidence=0.9, class_name="person", track_id=track_id,
    )


def _name_entity(reg: TrackedRegistry, track_id: int, name: str) -> None:
    """Set attrs['name'] on the entity whose internal track_key matches track_id."""
    for ent in reg.entities.values():
        if ent.track_key == track_id:
            ent.attrs["name"] = name
            return
    # Not found yet — entity may not have been spawned yet; silently skip.


def _step(reg: TrackedRegistry, game: FreezeGame, ts: float, dets: list[Detection],
          names: dict[int, str] | None = None) -> list:
    """Consume detections, assign names, then run the game trigger.

    ``names`` maps tracker track_id -> name string.  Names are set AFTER
    consume so entities exist, but BEFORE game.update so the game sees them.
    """
    reg.consume(dets, ts=ts)
    if names:
        for tid, name in names.items():
            _name_entity(reg, tid, name)
    return game.update(reg, ts)


# ---------------------------------------------------------------------------
# test_freeze_announces_after_settle
# ---------------------------------------------------------------------------


def test_freeze_announces_after_settle():
    """No freeze event before settle_s elapses; exactly one freeze after."""
    reg = _reg()
    game = FreezeGame(FAST)
    names = {1: "Alice", 2: "Bob"}

    # t=0.0 .. 0.15: eligible since t=0.0, but settle_s=0.2 not yet elapsed.
    for ts in (0.0, 0.05, 0.1, 0.15):
        dets = [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)]
        evs = _step(reg, game, ts, dets, names)
        assert not any(e.kind == "freeze" for e in evs), f"unexpected freeze at ts={ts}"

    # t=0.25 > 0.0 + 0.2 — should emit exactly one freeze.
    dets = [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)]
    evs = _step(reg, game, 0.25, dets, names)
    freeze_evs = [e for e in evs if e.kind == "freeze"]
    assert len(freeze_evs) == 1, f"expected 1 freeze event, got {freeze_evs}"


# ---------------------------------------------------------------------------
# test_catches_the_mover
# ---------------------------------------------------------------------------


def test_catches_the_mover():
    """Player B sweeps across a large distance; player A is static.
    freeze_result.name should be 'Bob'."""
    reg = _reg()
    game = FreezeGame(FAST)
    names = {1: "Alice", 2: "Bob"}

    # Settle phase.
    for ts in (0.0, 0.05, 0.1, 0.15):
        _step(reg, game, ts, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)

    # t=0.25 triggers freeze.
    evs = _step(reg, game, 0.25, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)
    assert any(e.kind == "freeze" for e in evs), "expected freeze at t=0.25"

    # Frozen window (freeze_window_s=0.5 → closes at ≥0.75).
    # Alice stays at (100,100); Bob sweeps x: 300 → 600 (300px clear motion).
    result_evs: list = []
    for i in range(12):
        frac = i / 11
        bob_x = 300.0 + frac * 300.0
        ts_f = 0.26 + i * 0.05   # 0.26 .. 0.81
        dets = [_person(1, 100.0, 100.0), _person(2, bob_x, 100.0)]
        evs = _step(reg, game, ts_f, dets, names)
        result_evs.extend(e for e in evs if e.kind == "freeze_result")
        if result_evs:
            break

    assert result_evs, "no freeze_result emitted"
    assert result_evs[0].name == "Bob", f"expected Bob, got {result_evs[0].name}"


# ---------------------------------------------------------------------------
# test_nobody_moved
# ---------------------------------------------------------------------------


def test_nobody_moved():
    """Both players remain still; freeze_result.name should be None."""
    reg = _reg()
    game = FreezeGame(FAST)
    names = {1: "Alice", 2: "Bob"}

    for ts in (0.0, 0.05, 0.1, 0.15):
        _step(reg, game, ts, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)

    evs = _step(reg, game, 0.25, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)
    assert any(e.kind == "freeze" for e in evs)

    result_evs: list = []
    for i in range(12):
        ts_f = 0.26 + i * 0.05
        dets = [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)]
        evs = _step(reg, game, ts_f, dets, names)
        result_evs.extend(e for e in evs if e.kind == "freeze_result")
        if result_evs:
            break

    assert result_evs, "no freeze_result emitted"
    assert result_evs[0].name is None, (
        f"expected None (nobody moved), got {result_evs[0].name}"
    )


# ---------------------------------------------------------------------------
# test_requires_min_players
# ---------------------------------------------------------------------------


def test_requires_min_players():
    """Only one named player present → no freeze ever emitted."""
    reg = _reg()
    game = FreezeGame(FAST)  # min_players=2

    all_evs: list = []
    for ts in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        dets = [_person(1, 100.0, 100.0)]   # only one player
        evs = _step(reg, game, ts, dets, {1: "Alice"})
        all_evs.extend(evs)

    assert not any(e.kind == "freeze" for e in all_evs), (
        "should not freeze with only 1 named player"
    )


# ---------------------------------------------------------------------------
# test_cooldown_blocks_replay
# ---------------------------------------------------------------------------


def test_cooldown_blocks_replay():
    """After a freeze_result, no new freeze fires until round_cooldown_s has elapsed."""
    reg = _reg()
    game = FreezeGame(FAST)  # round_cooldown_s=0.5
    names = {1: "Alice", 2: "Bob"}

    # Settle.
    for ts in (0.0, 0.05, 0.1, 0.15):
        _step(reg, game, ts, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)

    # Freeze fires at 0.25.
    evs = _step(reg, game, 0.25, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)
    assert any(e.kind == "freeze" for e in evs)

    # Run through frozen window.
    result_ts = 0.25
    for i in range(12):
        ts_f = 0.26 + i * 0.05
        evs = _step(reg, game, ts_f, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)
        if any(e.kind == "freeze_result" for e in evs):
            result_ts = ts_f
            break
    else:
        raise AssertionError("no freeze_result emitted")

    # During cooldown: no new freeze even though both players remain present.
    cooldown_end = result_ts + FAST.round_cooldown_s
    for i in range(4):
        ts_cool = result_ts + 0.05 * (i + 1)
        if ts_cool >= cooldown_end:
            break
        evs = _step(reg, game, ts_cool, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)],
                    names)
        assert not any(e.kind == "freeze" for e in evs), (
            f"freeze should not fire during cooldown at ts={ts_cool}"
        )

    # After cooldown + settle_s, a new freeze should eventually appear.
    post_start = cooldown_end + 0.01
    freeze_after: list = []
    for i in range(30):
        ts_post = post_start + i * 0.05
        evs = _step(reg, game, ts_post,
                    [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)
        freeze_after.extend(e for e in evs if e.kind == "freeze")
        if freeze_after:
            break

    assert freeze_after, "a new freeze should fire after cooldown + settle_s"


# ---------------------------------------------------------------------------
# test_leaving_player_is_caught
# ---------------------------------------------------------------------------


def test_leaving_player_is_caught():
    """A player present at freeze-start who is absent at window-end is caught.

    Bob's detections stop during the frozen window.  By window-end Bob has been
    gone for > 2.0s (the registry's default GONE_AFTER_S), so the registry has
    dropped him.  _build_present therefore cannot find him, and he gets
    ``score = inf`` → caught.
    """
    reg = _reg()
    game = FreezeGame(FAST)   # freeze_window_s=0.5
    names = {1: "Alice", 2: "Bob"}

    # Settle with large time steps to drive Bob's last_seen well into the past
    # before the window closes.  Bob stops appearing after t=0.25 (freeze trigger).
    # The frozen window closes at 0.25 + 0.5 = 0.75.
    # We then advance to t = 0.75 + 2.5 (well past GONE_AFTER_S=2.0 from last Bob det).
    for ts in (0.0, 0.05, 0.1, 0.15):
        _step(reg, game, ts, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)

    # Freeze fires.
    evs = _step(reg, game, 0.25, [_person(1, 100.0, 100.0), _person(2, 300.0, 100.0)], names)
    assert any(e.kind == "freeze" for e in evs)

    # During the frozen window: Alice is fed (stays still); Bob never appears again.
    # We collect enough samples for Alice BEFORE the window closes, and let Bob
    # coast until the registry drops him (GONE_AFTER_S=2.0 from last seen ts=0.25).
    # The window closes at ts > 0.25 + 0.5 = 0.75; Bob is GONE when ts > 0.25 + 2.0 = 2.25.
    # Strategy: feed Alice a few times inside the window, then jump past 2.25.

    # Feed Alice twice during the window so she has ≥2 samples.
    for ts_inner in (0.35, 0.50):
        _step(reg, game, ts_inner, [_person(1, 100.0, 100.0)], {1: "Alice"})

    # Now jump past both window-end (0.75) and Bob's GONE threshold (2.25).
    # At ts=2.5 the window will close AND Bob is already gone from the registry.
    result_evs: list = []
    for i in range(3):
        ts_f = 2.5 + i * 0.1
        dets = [_person(1, 100.0, 100.0)]
        evs = _step(reg, game, ts_f, dets, {1: "Alice"})
        result_evs.extend(e for e in evs if e.kind == "freeze_result")
        if result_evs:
            break

    assert result_evs, "no freeze_result emitted"
    assert result_evs[0].name == "Bob", (
        f"expected Bob (absent/gone at window end), got {result_evs[0].name}"
    )
