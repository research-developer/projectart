"""Freeze-Tag face game for scene mode.

State machine
-------------
IDLE → (min_players recognized for settle_s) → emit ``freeze`` → FROZEN
FROZEN → (freeze_window_s elapsed) → evaluate movement → emit ``freeze_result`` → COOLDOWN
COOLDOWN → (round_cooldown_s elapsed) → IDLE

Movement metric
---------------
During the FROZEN window we collect each rostered player's smoothed ``entity.center``
positions.  At window end we compute, per player:

    rms = sqrt( mean_over_samples( (x - mean_x)**2 + (y - mean_y)**2 ) )
    d   = mean bbox diagonal ( hypot(w, h) ) across samples
    score = rms / d   (normalised, scale-invariant)

The player with the highest score is the one who moved most.  If that score is
below ``move_tolerance`` nobody moved; otherwise that player is "caught".
A player who was on the roster at freeze-start but is absent at window-end is
given an infinite score (they broke the freeze by leaving entirely).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

from ..tracking.triggers import Event

log = logging.getLogger(__name__)

# ---- state labels -----------------------------------------------------------


class _State(Enum):
    IDLE = "idle"
    FROZEN = "frozen"
    COOLDOWN = "cooldown"


# ---- public config ----------------------------------------------------------


@dataclass(slots=True)
class FreezeConfig:
    """Tuning knobs for :class:`FreezeGame`."""

    players: tuple[str, ...] = ()
    """Specific required player names.  Empty tuple = any recognized people."""

    min_players: int = 2
    """Minimum number of recognized players needed to start a round."""

    settle_s: float = 2.0
    """All required players must be present continuously for this long before a round starts."""

    freeze_window_s: float = 10.0
    """Duration (seconds) to measure movement after the "Freeze!" event."""

    move_tolerance: float = 0.15
    """Normalised RMS movement below this threshold counts as "didn't move"."""

    round_cooldown_s: float = 8.0
    """Minimum gap between rounds."""

    class_name: str = "person"
    """YOLO class name to watch."""


# ---- game -------------------------------------------------------------------


class FreezeGame:
    """Conforms to the trigger protocol: ``update(registry, ts, frame_area=0.0) -> list[Event]``.

    Drop it into the scene's :class:`~projectart.tracking.triggers.TriggerEngine` and it
    drives ``freeze`` / ``freeze_result`` Events on the bus.
    """

    def __init__(self, config: FreezeConfig | None = None) -> None:
        self.cfg: FreezeConfig = config or FreezeConfig()
        self._state: _State = _State.IDLE
        # IDLE bookkeeping
        self._eligible_since: float | None = None
        # FROZEN bookkeeping
        self._freeze_start: float = 0.0
        self._roster: tuple[str, ...] = ()
        self._positions: dict[str, list[tuple[float, float]]] = {}
        self._diagonals: dict[str, list[float]] = {}
        # COOLDOWN bookkeeping
        self._cooldown_until: float = 0.0

    # ---- trigger protocol ---------------------------------------------------

    def update(self, registry, ts: float, frame_area: float = 0.0) -> list[Event]:
        """Called once per frame by TriggerEngine.  Returns at most one Event."""
        present = self._build_present(registry)

        if self._state is _State.IDLE:
            return self._tick_idle(present, ts)
        if self._state is _State.FROZEN:
            return self._tick_frozen(present, ts)
        # COOLDOWN
        return self._tick_cooldown(ts)

    # ---- per-state ticks ----------------------------------------------------

    def _tick_idle(
        self, present: dict[str, object], ts: float
    ) -> list[Event]:
        cfg = self.cfg
        eligible = len(present) >= cfg.min_players and (
            not cfg.players or all(p in present for p in cfg.players)
        )
        if not eligible:
            self._eligible_since = None
            return []
        if self._eligible_since is None:
            self._eligible_since = ts
            return []
        if ts - self._eligible_since >= cfg.settle_s:
            # Start the round.
            roster = tuple(cfg.players) if cfg.players else tuple(present.keys())
            self._roster = roster
            self._positions = {name: [] for name in roster}
            self._diagonals = {name: [] for name in roster}
            self._freeze_start = ts
            self._state = _State.FROZEN
            log.info("FreezeGame: FROZEN — roster=%s", roster)
            return [Event(kind="freeze", class_name=cfg.class_name, track_id=0, ts=ts)]
        return []

    def _tick_frozen(
        self, present: dict[str, object], ts: float
    ) -> list[Event]:
        cfg = self.cfg
        # Sample positions for rostered players who are visible.
        for name in self._roster:
            ent = present.get(name)
            if ent is not None:
                self._positions[name].append(ent.center)  # type: ignore[union-attr]
                bb = ent.last_bbox  # type: ignore[union-attr]
                self._diagonals[name].append(math.hypot(bb.w, bb.h))

        if ts - self._freeze_start < cfg.freeze_window_s:
            return []

        # Window closed — evaluate.
        caught = self._evaluate(present)
        self._state = _State.COOLDOWN
        self._cooldown_until = ts + cfg.round_cooldown_s
        self._roster = ()
        self._positions = {}
        self._diagonals = {}
        log.info("FreezeGame: freeze_result — caught=%s", caught)
        return [
            Event(
                kind="freeze_result",
                class_name=cfg.class_name,
                track_id=0,
                name=caught,
                ts=ts,
            )
        ]

    def _tick_cooldown(self, ts: float) -> list[Event]:
        if ts >= self._cooldown_until:
            self._state = _State.IDLE
            self._eligible_since = None
            log.debug("FreezeGame: back to IDLE")
        return []

    # ---- helpers ------------------------------------------------------------

    def _build_present(self, registry) -> dict[str, object]:
        """Build name→entity map from confirmed entities of the configured class."""
        seen: dict[str, object] = {}
        for ent in registry.confirmed():
            if ent.class_name != self.cfg.class_name:
                continue
            name = ent.attrs.get("name")
            if not name:
                continue
            if name not in seen:
                seen[name] = ent
        return seen

    def _evaluate(self, present: dict[str, object]) -> str | None:
        """Return the name of the player who moved most, or None if nobody moved."""
        scores: dict[str, float] = {}
        for name in self._roster:
            if name not in present:
                scores[name] = float("inf")
                continue
            samples = self._positions.get(name, [])
            if len(samples) < 2:
                scores[name] = float("inf")
                continue
            n = len(samples)
            mean_x = sum(x for x, _ in samples) / n
            mean_y = sum(y for _, y in samples) / n
            var = sum((x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in samples) / n
            rms = math.sqrt(var)
            diags = self._diagonals.get(name, [])
            d = sum(diags) / len(diags) if diags else 0.0
            scores[name] = rms / d if d > 0 else 0.0

        if not scores:
            return None
        caught_name = max(scores, key=lambda k: scores[k])
        if scores[caught_name] < self.cfg.move_tolerance:
            return None
        return caught_name
