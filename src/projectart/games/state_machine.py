"""Guarded state machine base for projectart games.

Design
------
Subclasses declare three class-level attributes::

    _STATES: type[Enum]
    _INITIAL: Enum
    _TRANSITIONS: dict[Enum, set[Enum]]   # from_state -> {allowed to_states}

Every state change goes through :meth:`_to`, which validates the transition
against the table.

strict=True (recommended for tests)
    An illegal transition raises :exc:`IllegalTransitionError` so a regression
    fails loudly.

strict=False (default, used in the live game)
    An illegal transition is logged at WARNING and silently ignored, so a
    running game never crashes on an unforeseen edge.

``dst == current`` is a no-op in both modes: it returns ``False`` and nothing
is appended to the transition log.
"""
from __future__ import annotations

import logging
from enum import Enum

log = logging.getLogger(__name__)


class IllegalTransitionError(RuntimeError):
    """Raised (only in strict mode) when a transition not in the allowed table is attempted."""


class StateMachine:
    """Base for the games' state machines.

    Subclasses must set three class attributes:

    * ``_STATES``      — the :class:`~enum.Enum` type for all states.
    * ``_INITIAL``     — the starting state (a member of ``_STATES``).
    * ``_TRANSITIONS`` — a ``dict[Enum, set[Enum]]`` mapping each state to the
      set of states it may transition to.

    Every state change must go through :meth:`_to` so that the guard and the
    transition log stay consistent.
    """

    _STATES: type[Enum]
    _INITIAL: Enum
    _TRANSITIONS: dict

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self._state = self._INITIAL
        self.transitions: list[tuple] = []  # (ts, from_state, to_state) for each ACTUAL transition

    @property
    def state(self) -> Enum:
        """Current state (read-only)."""
        return self._state

    def _allowed(self, src: Enum, dst: Enum) -> bool:
        return dst in self._TRANSITIONS.get(src, set())

    def _to(self, dst: Enum, *, ts: float | None = None) -> bool:
        """Attempt a transition to *dst*.

        Parameters
        ----------
        dst:
            Target state.
        ts:
            Timestamp to record in the transition log (may be ``None``).

        Returns
        -------
        bool
            ``True`` when the state actually changed, ``False`` otherwise
            (no-op or rejected illegal transition).

        Raises
        ------
        IllegalTransitionError
            Only when ``self.strict is True`` and the transition is not in
            ``_TRANSITIONS``.
        """
        src = self._state
        if dst is src:
            return False  # no-op — not logged

        if not self._allowed(src, dst):
            msg = (
                f"Illegal transition {src.name} → {dst.name} "
                f"in {type(self).__name__}"
            )
            if self.strict:
                raise IllegalTransitionError(msg)
            log.warning("%s", msg)
            return False

        self._state = dst
        self.transitions.append((ts, src, dst))
        return True
