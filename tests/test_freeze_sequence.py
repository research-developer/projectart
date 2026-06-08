"""Sequence-level tests for FreezeGame via the GameDriver harness.

These tests focus on the ORDER of emitted Events and the state-machine
transition log, not just final outcomes.  The existing outcome tests live in
test_freeze_tag.py and continue to run unchanged.
"""
from __future__ import annotations

import pytest
from support.game_driver import GameDriver, person

from projectart.games.freeze_tag import FreezeConfig, FreezeGame
from projectart.games.state_machine import IllegalTransitionError

# Fast config (matching the existing test_freeze_tag.py FAST constant).
FAST = FreezeConfig(
    settle_s=0.2,
    freeze_window_s=0.5,
    round_cooldown_s=0.5,
    move_tolerance=0.1,
)

_NAMES = {1: "Alice", 2: "Bob"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _both_present(bob_x: float = 300.0) -> list:
    return [person(1, 100.0, 100.0), person(2, bob_x, 100.0)]


def _drive_to_freeze(driver: GameDriver) -> float:
    """Advance through the settle phase and return the timestamp of the freeze event."""
    for ts in (0.0, 0.05, 0.1, 0.15):
        driver.step(ts, _both_present(), _NAMES)
    evs = driver.step(0.25, _both_present(), _NAMES)
    assert any(e.kind == "freeze" for e in evs), "settle phase did not produce a freeze event"
    return 0.25


def _drive_frozen_window(
    driver: GameDriver,
    freeze_ts: float,
    *,
    bob_sweeps: bool = False,
) -> float:
    """Drive through the frozen window; return the timestamp of the freeze_result event."""
    result_ts = None
    for i in range(15):
        frac = i / 14
        bob_x = 300.0 + (frac * 300.0 if bob_sweeps else 0.0)
        ts_f = freeze_ts + 0.05 + i * 0.05
        evs = driver.step(ts_f, _both_present(bob_x), _NAMES)
        if any(e.kind == "freeze_result" for e in evs):
            result_ts = ts_f
            break
    assert result_ts is not None, "no freeze_result emitted during frozen window"
    return result_ts


def _drive_cooldown(driver: GameDriver, result_ts: float) -> float:
    """Drive through the cooldown phase; return the first ts where IDLE is re-entered."""
    rearm_ts = result_ts + FAST.round_cooldown_s + 0.01
    for i in range(20):
        ts_c = result_ts + 0.05 * (i + 1)
        driver.step(ts_c, _both_present(), _NAMES)
        if ts_c >= rearm_ts:
            return ts_c
    return rearm_ts


# ---------------------------------------------------------------------------
# test_full_round_sequence
# ---------------------------------------------------------------------------


def test_full_round_sequence():
    """Drive a complete round and verify both the event kind sequence and the
    state-machine transition log."""
    driver = GameDriver(FreezeGame(FAST, strict=True))

    # Round 1 ----------------------------------------------------------------
    freeze_ts = _drive_to_freeze(driver)
    result_ts = _drive_frozen_window(driver, freeze_ts, bob_sweeps=True)

    # Now drive through cooldown + re-settle for a second freeze.
    rearm_ts = result_ts + FAST.round_cooldown_s + 0.01
    second_freeze_ts = None
    for i in range(40):
        ts_post = rearm_ts + i * 0.05
        evs = driver.step(ts_post, _both_present(), _NAMES)
        if any(e.kind == "freeze" for e in evs):
            second_freeze_ts = ts_post
            break

    assert second_freeze_ts is not None, "game did not re-arm after cooldown"

    # Event sequence: at minimum freeze, freeze_result, freeze in that order.
    kinds = driver.kinds()
    freeze_indices = [i for i, k in enumerate(kinds) if k == "freeze"]
    result_indices = [i for i, k in enumerate(kinds) if k == "freeze_result"]

    assert len(freeze_indices) >= 2, f"expected >= 2 freezes, got {freeze_indices}"
    assert len(result_indices) >= 1, f"expected >= 1 freeze_result, got {result_indices}"

    # First freeze_result comes after the first freeze.
    assert freeze_indices[0] < result_indices[0], (
        "first freeze_result must come after first freeze"
    )
    # Second freeze comes after the first freeze_result.
    assert result_indices[0] < freeze_indices[1], (
        "second freeze must come after first freeze_result"
    )

    # Transition log: IDLE→FROZEN→COOLDOWN→IDLE→FROZEN…
    pairs = [(src.name, dst.name) for _, src, dst in driver.transitions]
    # There must be at least the first round's three transitions.
    assert len(pairs) >= 3
    assert pairs[0] == ("IDLE", "FROZEN")
    assert pairs[1] == ("FROZEN", "COOLDOWN")
    assert pairs[2] == ("COOLDOWN", "IDLE")
    if len(pairs) >= 4:
        assert pairs[3] == ("IDLE", "FROZEN")


# ---------------------------------------------------------------------------
# test_no_result_without_preceding_freeze
# ---------------------------------------------------------------------------


def test_no_result_without_preceding_freeze():
    """Invariant: every freeze_result is preceded by a freeze with no freeze_result
    between them (no orphan results)."""
    driver = GameDriver(FreezeGame(FAST, strict=True))

    freeze_ts = _drive_to_freeze(driver)
    _drive_frozen_window(driver, freeze_ts, bob_sweeps=False)

    kinds = driver.kinds()
    for result_idx, k in enumerate(kinds):
        if k != "freeze_result":
            continue
        # Find the most recent freeze before this result_idx.
        preceding_freeze = None
        for j in range(result_idx - 1, -1, -1):
            if kinds[j] == "freeze":
                preceding_freeze = j
                break
            if kinds[j] == "freeze_result":
                # Another result before a freeze — invariant broken.
                pytest.fail(
                    f"freeze_result at index {result_idx} is not preceded by a freeze "
                    f"(found another freeze_result at index {j})"
                )
        assert preceding_freeze is not None, (
            f"freeze_result at index {result_idx} has no preceding freeze event"
        )


# ---------------------------------------------------------------------------
# test_no_double_freeze
# ---------------------------------------------------------------------------


def test_no_double_freeze():
    """No second freeze fires while the game is FROZEN or in COOLDOWN (one per round)."""
    driver = GameDriver(FreezeGame(FAST, strict=True))

    freeze_ts = _drive_to_freeze(driver)
    # Drive well past the frozen window but check for duplicates within the window.
    for i in range(15):
        ts_f = freeze_ts + 0.05 + i * 0.05
        driver.step(ts_f, _both_present(), _NAMES)

    # Count freeze events that happened BEFORE any freeze_result.
    kinds = driver.kinds()
    first_result = next((i for i, k in enumerate(kinds) if k == "freeze_result"), len(kinds))
    freezes_before_result = [k for k in kinds[:first_result] if k == "freeze"]
    assert len(freezes_before_result) == 1, (
        f"expected exactly 1 freeze before first freeze_result, "
        f"got {len(freezes_before_result)}: {kinds}"
    )

    # Also confirm no freeze fires during cooldown.
    result_ts = None
    for i, k in enumerate(kinds):
        if k == "freeze_result":
            result_ts = driver.events[i].ts
            break
    assert result_ts is not None

    cooldown_end = result_ts + FAST.round_cooldown_s
    for i in range(8):
        ts_cool = result_ts + 0.05 * (i + 1)
        if ts_cool >= cooldown_end:
            break
        evs = driver.step(ts_cool, _both_present(), _NAMES)
        assert not any(e.kind == "freeze" for e in evs), (
            f"unexpected freeze during cooldown at ts={ts_cool}"
        )


# ---------------------------------------------------------------------------
# test_min_players_gate
# ---------------------------------------------------------------------------


def test_min_players_gate():
    """With only one named player present, no events are emitted and the game
    never leaves IDLE (the transition log stays empty)."""
    driver = GameDriver(FreezeGame(FAST, strict=True))

    for ts in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        evs = driver.step(ts, [person(1, 100.0, 100.0)], {1: "Alice"})
        assert evs == [], f"unexpected events with one player at ts={ts}: {evs}"

    assert driver.kinds() == [], "no events expected"
    assert driver.transitions == [], "should never leave IDLE"


# ---------------------------------------------------------------------------
# test_strict_survives_full_round
# ---------------------------------------------------------------------------


def test_strict_survives_full_round():
    """A strict FreezeGame runs a complete round without raising IllegalTransitionError.

    This confirms that the real IDLE→FROZEN→COOLDOWN→IDLE transition sequence is
    fully contained in _TRANSITIONS, i.e. the strict game never hits an illegal edge.
    """
    driver = GameDriver(FreezeGame(FAST, strict=True))

    try:
        freeze_ts = _drive_to_freeze(driver)
        result_ts = _drive_frozen_window(driver, freeze_ts, bob_sweeps=True)
        _drive_cooldown(driver, result_ts)
    except IllegalTransitionError as exc:
        pytest.fail(f"strict FreezeGame raised IllegalTransitionError: {exc}")

    # Sanity: we got the expected events.
    assert "freeze" in driver.kinds()
    assert "freeze_result" in driver.kinds()


# ---------------------------------------------------------------------------
# test_freeze_config_validation
# ---------------------------------------------------------------------------


def test_freeze_config_validation_min_players():
    with pytest.raises(ValueError, match="min_players"):
        FreezeConfig(min_players=0)


def test_freeze_config_validation_settle_s():
    with pytest.raises(ValueError, match="settle_s"):
        FreezeConfig(settle_s=-0.1)


def test_freeze_config_validation_freeze_window_s():
    with pytest.raises(ValueError, match="freeze_window_s"):
        FreezeConfig(freeze_window_s=0.0)


def test_freeze_config_validation_round_cooldown_s():
    with pytest.raises(ValueError, match="round_cooldown_s"):
        FreezeConfig(round_cooldown_s=-1.0)


def test_freeze_config_validation_move_tolerance():
    with pytest.raises(ValueError, match="move_tolerance"):
        FreezeConfig(move_tolerance=-0.01)


def test_freeze_config_validation_players_not_tuple():
    with pytest.raises((ValueError, TypeError)):
        FreezeConfig(players=["Alice"])  # type: ignore[arg-type]


def test_freeze_config_validation_players_empty_string():
    with pytest.raises(ValueError, match="players"):
        FreezeConfig(players=("Alice", ""))


def test_freeze_config_valid_empty_players():
    """Empty players tuple is valid."""
    cfg = FreezeConfig(players=())
    assert cfg.players == ()


def test_freeze_config_valid_named_players():
    """Non-empty players tuple of non-empty strings is valid."""
    cfg = FreezeConfig(players=("Alice", "Bob"))
    assert cfg.players == ("Alice", "Bob")
