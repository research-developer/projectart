"""Lightweight event bus for TrackedEntity behaviors.

Simple synchronous publish/subscribe. Keeps the tracking module pure
(no UI, no audio) — those layers attach by name.

    bus = BehaviorBus()
    bus.on("cat.appeared", lambda *, entity: play_meow())
    bus.on("cat.left", lambda *, entity: stop_meow())

Events are dispatched in subscription order; exceptions in one handler
don't stop the rest.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable

log = logging.getLogger(__name__)

Handler = Callable[..., None]


class BehaviorBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def on(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    def off(self, topic: str, handler: Handler) -> None:
        try:
            self._subs[topic].remove(handler)
        except ValueError:
            pass

    def emit(self, topic: str, **kwargs) -> None:
        # `topic` (not `event`) so callers can forward an `event=` kwarg to
        # handlers without colliding with this parameter — see scene.py.
        for h in list(self._subs.get(topic, ())):
            try:
                h(**kwargs)
            except Exception:
                log.exception("handler for %r raised", topic)
