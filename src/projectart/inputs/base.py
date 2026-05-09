from __future__ import annotations

from typing import Protocol

from ..server.protocol import HudAnchorEvent, PointerEvent  # noqa: F401  (re-export)


class InputSource(Protocol):
    """All input modes (mouse, gloves, wand, androidtv) implement this contract.

    `run()` is an async coroutine that owns the input lifecycle and drives the
    server's `broadcast()` continuously until cancellation."""

    async def run(self) -> None: ...
