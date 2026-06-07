#!/usr/bin/env python3
"""Generate Freeze-Tag game audio clips with supertonic.

Writes WAV files into ~/.projectart/audio/freeze (created if absent).  Two
generic clips are always produced (``freeze_start`` and ``freeze_clear``); for
each ``--name`` given a name-specific "caught" clip is also generated.

    python tools/gen_freeze_audio.py
    python tools/gen_freeze_audio.py --name "Samaya" --name "Preston Temple"
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_VOICES = ["F1", "M1", "F3"]

GENERIC_CLIPS: dict[str, str] = {
    "freeze_start": "Freeze! Everybody hold still!",
    "freeze_clear": "Nobody moved! You all win!",
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _first(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate Freeze-Tag voice clips via supertonic TTS."
    )
    ap.add_argument("--name", action="append", dest="names", default=[],
                    help="player name to generate caught-clip for (repeatable)")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                    help="comma-separated voice codes (default: F1,M1,F3)")
    ap.add_argument("--steps", type=int, default=10,
                    help="supertonic diffusion steps (default: 10)")
    ap.add_argument("--out-dir", default="~/.projectart/audio/freeze",
                    help="output directory (default: ~/.projectart/audio/freeze)")
    args = ap.parse_args()

    if shutil.which("supertonic") is None:
        print("error: 'supertonic' not on PATH", file=sys.stderr)
        return 1

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    made = 0

    # Generic clips (always generated).
    for situation, text in GENERIC_CLIPS.items():
        for voice in voices:
            dest = out / f"{situation}__{voice}.wav"
            r = subprocess.run(
                ["supertonic", "tts", text, "-o", str(dest), "--voice", voice,
                 "--steps", str(args.steps)],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not dest.exists():
                print(f"FAIL {dest.name}: {r.stderr.strip()[:160]}", file=sys.stderr)
                continue
            made += 1
            print(f"  {dest.name}  ({text!r})")

    # Per-player "caught" clips.
    for name in args.names:
        text = f"{_first(name)}, you moved! You're it!"
        situation = f"freeze_caught_{_slug(name)}"
        for voice in voices:
            dest = out / f"{situation}__{voice}.wav"
            r = subprocess.run(
                ["supertonic", "tts", text, "-o", str(dest), "--voice", voice,
                 "--steps", str(args.steps)],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not dest.exists():
                print(f"FAIL {dest.name}: {r.stderr.strip()[:160]}", file=sys.stderr)
                continue
            made += 1
            print(f"  {dest.name}  ({text!r})")

    print(f"\nwrote {made} clips to {out}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
