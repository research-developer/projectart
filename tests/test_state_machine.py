"""Unit tests for the StateMachine base in isolation.

We use a tiny toy 3-state machine (A → B → C → A) to exercise the guarded
transition logic without any game-level concerns.
"""
from __future__ import annotations

import logging
from enum import Enum

import pytest

from projectart.games.state_machine import IllegalTransitionError, StateMachine

# ---------------------------------------------------------------------------
# Toy state machine for testing
# ---------------------------------------------------------------------------


class _S(Enum):
    A = "a"
    B = "b"
    C = "c"


class ToyMachine(StateMachine):
    _STATES = _S
    _INITIAL = _S.A
    _TRANSITIONS = {
        _S.A: {_S.B},
        _S.B: {_S.C},
        _S.C: {_S.A},
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_legal_transition_changes_state():
    """A legal transition updates state and appends one entry to the log."""
    m = ToyMachine()
    result = m._to(_S.B, ts=1.0)
    assert result is True
    assert m.state is _S.B
    assert len(m.transitions) == 1
    ts, src, dst = m.transitions[0]
    assert ts == 1.0
    assert src is _S.A
    assert dst is _S.B


def test_noop_same_state():
    """Transitioning to the current state is a no-op; nothing is logged."""
    m = ToyMachine()
    result = m._to(_S.A, ts=0.5)
    assert result is False
    assert m.state is _S.A
    assert m.transitions == []


def test_illegal_transition_strict_raises():
    """strict=True: an illegal transition raises IllegalTransitionError."""
    m = ToyMachine(strict=True)
    with pytest.raises(IllegalTransitionError):
        m._to(_S.C, ts=2.0)  # A → C is not in the table
    # State must not have changed.
    assert m.state is _S.A
    assert m.transitions == []


def test_illegal_transition_lenient_returns_false(caplog):
    """strict=False (default): illegal transition logs a WARNING and returns False."""
    m = ToyMachine(strict=False)
    with caplog.at_level(logging.WARNING, logger="projectart.games.state_machine"):
        result = m._to(_S.C, ts=3.0)  # A → C is not in the table
    assert result is False
    assert m.state is _S.A
    assert m.transitions == []
    # A warning must have been emitted.
    assert any("Illegal transition" in r.message for r in caplog.records)


def test_multiple_legal_transitions_log_sequence():
    """Chained legal transitions accumulate an ordered log."""
    m = ToyMachine()
    m._to(_S.B, ts=1.0)
    m._to(_S.C, ts=2.0)
    m._to(_S.A, ts=3.0)
    assert m.state is _S.A
    assert [(src.name, dst.name) for _, src, dst in m.transitions] == [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
    ]


def test_state_property_is_readonly_via_to():
    """The .state property always reflects the internal _state."""
    m = ToyMachine()
    assert m.state is _S.A
    m._to(_S.B)
    assert m.state is _S.B


def test_initial_state_from_class_attr():
    """_INITIAL drives the starting state in every fresh instance."""
    m = ToyMachine()
    assert m.state is _S.A
    assert m.transitions == []
