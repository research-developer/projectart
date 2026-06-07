#!/usr/bin/env python3
"""Generate per-person reaction clips (greeting + name-aware cat-intersect) with supertonic.

Writes ``<situation>_<slug>__<voice>.wav`` into ~/.projectart/audio/people, where
``slug`` is the person's full name lowercased with spaces->underscores (matching the
face-gallery names). The CatAudioPlayer searches ~/.projectart/audio recursively, so
these are found alongside the generic cat clips. Clips are not committed.

    python tools/gen_person_audio.py --name "Samaya" --name "Preston Temple"
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATES = {
    "greet": "Hi {first}!",
    "farewell": "Bye {first}, see you later!",
    "intersect_look": "Look at {first}'s cute little meow meow.",
    "intersect_benice": "{first}, be nice to the meow meow.",
}
DEFAULT_VOICES = ["F1", "M1", "F3"]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _first(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True, dest="names")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out-dir", default="~/.projectart/audio/people")
    args = ap.parse_args()

    if shutil.which("supertonic") is None:
        print("error: 'supertonic' not on PATH", file=sys.stderr)
        return 1
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    made = 0
    for name in args.names:
        for situ, tmpl in TEMPLATES.items():
            text = tmpl.format(first=_first(name))
            for voice in voices:
                dest = out / f"{situ}_{_slug(name)}__{voice}.wav"
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
