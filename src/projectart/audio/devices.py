"""Audio output device discovery + name->index resolution (via sounddevice).

Used to let the user pick a playback device by name (e.g. "Echo Show-E99").
Degrades gracefully to an empty list if sounddevice isn't installed.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def list_output_devices() -> list[tuple[int, str]]:
    """(index, name) for every device with output channels (empty if sounddevice
    is unavailable)."""
    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover - environment dependent
        log.warning("sounddevice unavailable (%s); cannot enumerate devices", e)
        return []
    out: list[tuple[int, str]] = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            out.append((i, str(d["name"])))
    return out


def find_device(name: str) -> int | None:
    """Index of the first output device whose name contains `name`
    (case-insensitive), or None."""
    if not name:
        return None
    needle = name.lower()
    for idx, dev_name in list_output_devices():
        if needle in dev_name.lower():
            return idx
    return None


def format_output_devices() -> str:
    devs = list_output_devices()
    if not devs:
        return "No output audio devices found (is sounddevice installed?)."
    return "Output audio devices:\n" + "\n".join(f"  [{i}] {n}" for i, n in devs)
