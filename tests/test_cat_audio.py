"""Tests for the cat audio player (mapping + cooldown), with fake playback."""
from __future__ import annotations

import random

from projectart.audio.cat_audio import CatAudioConfig, CatAudioPlayer
from projectart.tracking.triggers import Event


def _player(tmp_path, **kw):
    played: list = []
    cfg = CatAudioConfig(audio_dir=tmp_path, **kw)
    p = CatAudioPlayer(config=cfg, play=played.append, rng=random.Random(0))
    return p, played


def _wavs(tmp_path, *names):
    for n in names:
        (tmp_path / n).touch()


def test_situation_for_appear_sizes(tmp_path):
    p, _ = _player(tmp_path)
    assert p.situation_for(Event("appear", "cat", 1, size="far")) == "cat_appear_far"
    assert p.situation_for(Event("appear", "cat", 1, size="near")) == "cat_appear_near"


def test_situation_for_disappear_intersect_and_none(tmp_path):
    p, _ = _player(tmp_path)
    assert p.situation_for(Event("disappear", "cat", 1)) == "cat_leave"
    s = p.situation_for(Event("intersect", "person", 1, other_class="cat", other_track_id=2))
    assert s in ("intersect_look", "intersect_benice")
    assert p.situation_for(Event("appear", "dog", 1)) is None


def test_clip_for(tmp_path):
    p, _ = _player(tmp_path)
    assert p.clip_for("cat_appear_far") is None
    _wavs(tmp_path, "cat_appear_far__F1.wav", "cat_appear_far__M1.wav")
    clip = p.clip_for("cat_appear_far")
    assert clip is not None and clip.name.startswith("cat_appear_far__")


def test_on_event_plays_and_respects_cooldown(tmp_path):
    _wavs(tmp_path, "cat_appear_far__F1.wav")
    p, played = _player(tmp_path, cooldown_s=1.0)
    assert p.on_event(Event("appear", "cat", 1, size="far", ts=0.0)) is not None
    assert len(played) == 1
    assert p.on_event(Event("appear", "cat", 2, size="far", ts=0.5)) is None   # within cooldown
    assert len(played) == 1
    assert p.on_event(Event("appear", "cat", 3, size="far", ts=1.5)) is not None  # after cooldown
    assert len(played) == 2


def test_disabled_plays_nothing(tmp_path):
    _wavs(tmp_path, "cat_appear_far__F1.wav")
    p, played = _player(tmp_path, enabled=False)
    assert p.on_event(Event("appear", "cat", 1, size="far", ts=0.0)) is None
    assert played == []


def test_missing_clip_returns_none(tmp_path):
    p, played = _player(tmp_path)  # empty audio dir
    assert p.on_event(Event("appear", "cat", 1, size="far", ts=0.0)) is None
    assert played == []


def test_unknown_device_falls_back_to_afplay(tmp_path, monkeypatch):
    from projectart.audio import cat_audio, devices
    monkeypatch.setattr(devices, "list_output_devices", lambda: [])  # find_device -> None
    p = cat_audio.CatAudioPlayer(cat_audio.CatAudioConfig(audio_dir=tmp_path, device="ghost"))
    assert p._play is cat_audio.afplay


def test_known_device_routes_off_afplay(tmp_path, monkeypatch):
    from projectart.audio import cat_audio, devices
    monkeypatch.setattr(devices, "list_output_devices", lambda: [(3, "Echo Show-E99")])
    p = cat_audio.CatAudioPlayer(cat_audio.CatAudioConfig(audio_dir=tmp_path, device="echo"))
    assert p._play is not cat_audio.afplay
