#!/usr/bin/env python3
"""Generate cat-reaction voice clips with the supertonic TTS CLI.

Writes one WAV per (situation, voice) into the output dir (default
``~/.projectart/audio/cat``). The cat-audio behavior picks a clip per fired
Event, choosing randomly among a situation's voice variants for variety.

Requires ``supertonic`` on PATH (symlink to the conda ``tts`` env). The clips
themselves are NOT committed — regenerate with this script:

    python tools/gen_cat_audio.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# situation key -> spoken line. Keys are the contract the cat-audio behavior maps
# Events onto (see audio behavior). Size + intersect variants are baked into the keys.
SITUATIONS = {
    "cat_appear_far": "Come here, meow meow!",
    "cat_appear_near": "Easy there, meow meow, I'm not your food.",
    "cat_leave": "Aww, bye bye, meow meow.",
    "intersect_look": "Look at your cute little meow meow.",
    "intersect_benice": "Be nice to the meow meow.",
}
DEFAULT_VOICES = ["F1", "M1", "F3"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate cat-reaction voice clips (supertonic)")
    ap.add_argument("--out-dir", default="~/.projectart/audio/cat")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES), help="comma-separated voice ids")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    if shutil.which("supertonic") is None:
        print("error: 'supertonic' not on PATH (symlink the conda tts env binary)",
              file=sys.stderr)
        return 1

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    made = []
    for situ, text in SITUATIONS.items():
        for voice in voices:
            dest = out / f"{situ}__{voice}.wav"
            cmd = ["supertonic", "tts", text, "-o", str(dest),
                   "--voice", voice, "--steps", str(args.steps)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not dest.exists():
                print(f"FAIL {dest.name}: {r.stderr.strip()[:200]}", file=sys.stderr)
                continue
            made.append(dest)
            print(f"  {dest.name}  ({text!r})")

    print(f"\nwrote {len(made)} clips to {out}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
