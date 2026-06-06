"""Tests for audio output device discovery / name resolution."""
from __future__ import annotations

from projectart.audio import devices


def test_find_device_substring_case_insensitive(monkeypatch):
    monkeypatch.setattr(devices, "list_output_devices",
                        lambda: [(0, "Echo Show-E99"), (6, "MacBook Pro Speakers")])
    assert devices.find_device("echo show") == 0
    assert devices.find_device("MacBook") == 6
    assert devices.find_device("nope") is None
    assert devices.find_device("") is None


def test_format_output_devices_lists(monkeypatch):
    monkeypatch.setattr(devices, "list_output_devices", lambda: [(0, "Echo Show-E99")])
    s = devices.format_output_devices()
    assert "[0]" in s and "Echo Show-E99" in s


def test_format_output_devices_empty(monkeypatch):
    monkeypatch.setattr(devices, "list_output_devices", lambda: [])
    assert "No output audio devices" in devices.format_output_devices()
