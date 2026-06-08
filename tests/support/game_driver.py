"""Thin test harness for any trigger/game that implements
``update(registry, ts) -> list[Event]``.

Usage::

    from support.game_driver import GameDriver, person

    driver = GameDriver(game)
    evs = driver.step(ts=0.0, dets=[person(1, 100.0, 100.0)], names={1: "Alice"})
    assert driver.kinds() == ["freeze"]
    assert driver.transitions[0] == (...)

``person(track_id, cx, cy, w=60.0, h=120.0)`` returns a ``Detection`` typed
as class_id=0, class_name="person", confidence=0.9, with an explicit track_id
so the registry uses its stable-id association path (matching the existing
freeze tests).
"""
from __future__ import annotations

from projectart.detection.yolo_dots import Detection
from projectart.tracking import TrackedRegistry
from projectart.tracking.builtins import Cat, Person
from projectart.tracking.triggers import Event


def person(
    track_id: int,
    cx: float,
    cy: float,
    w: float = 60.0,
    h: float = 120.0,
) -> Detection:
    """Return a person Detection with an explicit track_id."""
    return Detection(
        class_id=0,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        confidence=0.9,
        class_name="person",
        track_id=track_id,
    )


class GameDriver:
    """Script a timeline through any trigger/game, recording the full Event stream.

    Parameters
    ----------
    game:
        Any object with ``update(registry, ts) -> list[Event]``.
    entity_types:
        Entity classes to register; defaults to ``[Cat, Person]``.
    """

    def __init__(self, game, *, entity_types: list | None = None) -> None:
        self.game = game
        self.reg = TrackedRegistry(entity_types=entity_types or [Cat, Person])
        self.events: list[Event] = []

    def step(
        self,
        ts: float,
        dets: list[Detection],
        names: dict[int, str] | None = None,
    ) -> list[Event]:
        """Advance the simulation by one tick.

        1. ``registry.consume(dets, ts=ts)``
        2. Set ``attrs["name"]`` on any entity whose ``track_key`` matches a
           key in *names*.
        3. Call ``game.update(registry, ts)`` and append results to
           ``self.events``.

        Parameters
        ----------
        ts:
            Simulation timestamp.
        dets:
            List of :class:`~projectart.detection.yolo_dots.Detection` objects.
        names:
            Optional mapping of tracker track_id → name string.  Names are set
            **after** consume (so entities exist) and **before** game.update
            (so the game sees them).

        Returns
        -------
        list[Event]
            Events emitted this tick.
        """
        self.reg.consume(dets, ts=ts)
        if names:
            for tid, name in names.items():
                self._set_name(tid, name)
        evs = self.game.update(self.reg, ts)
        self.events.extend(evs)
        return evs

    def _set_name(self, track_id: int, name: str) -> None:
        """Set attrs['name'] on the entity whose internal track_key equals track_id."""
        for ent in self.reg.entities.values():
            if ent.track_key == track_id:
                ent.attrs["name"] = name
                return

    def kinds(self) -> list[str]:
        """Return ``[e.kind for e in self.events]`` — the ordered kind sequence."""
        return [e.kind for e in self.events]

    def of_kind(self, kind: str) -> list[Event]:
        """Return all accumulated events whose ``kind`` matches."""
        return [e for e in self.events if e.kind == kind]

    @property
    def transitions(self) -> list[tuple]:
        """The state-machine transition log from the game, or ``[]`` if unavailable."""
        return getattr(self.game, "transitions", [])
