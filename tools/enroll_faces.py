#!/usr/bin/env python3
"""Enroll people into the face gallery from Apple Photos — fully local.

For each named person (from Photos' People), scan recent photos and embed THAT
person's tagged face: Photos already records each face's region + name, so we
pick the YuNet detection nearest the person's tagged-face centre and embed it
(SFace). This handles multi-face photos and avoids enrolling the wrong person.

Images are read in place (not copied); only the (non-reversible) embeddings are
stored in ~/.projectart/faces/gallery.npz.

    python tools/enroll_faces.py --person "Samaya" --person "Preston Temple" \
        --months 12 --per-person 40
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _oriented_bgr(path: str, tmp: Path):
    """Load as EXIF-oriented BGR so pixel coords match Photos' face regions;
    fall back to sips (HEIC/unsupported)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        dest = tmp / "pa_conv.jpg"
        try:
            subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", str(dest)],
                           capture_output=True, check=True)
            im = ImageOps.exif_transpose(Image.open(dest)).convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
        finally:
            dest.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", action="append", required=True, dest="persons")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--per-person", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=250, help="max recent photos to scan")
    ap.add_argument("--min-quality", type=float, default=0.15)
    ap.add_argument("--append", action="store_true",
                    help="merge into the existing gallery (replace only the persons enrolled now)")
    ap.add_argument("--out", default="~/.projectart/faces/gallery.npz")
    args = ap.parse_args()

    try:
        import osxphotos
    except Exception as e:  # pragma: no cover
        print("need osxphotos (pip install osxphotos):", e, file=sys.stderr)
        return 1

    from projectart.detection.faces import FaceGallery, FaceRecognizer

    rec = FaceRecognizer()
    out = Path(args.out).expanduser()
    gallery = FaceGallery()
    if args.append and out.exists():
        gallery = FaceGallery.load(out)
        print(f"appending to existing gallery: {{{', '.join(gallery.names())}}}")
    print("opening Photos library (may take a moment)...")
    db = osxphotos.PhotosDB()
    cutoff = datetime.datetime.now().astimezone() - datetime.timedelta(days=args.months * 30)
    tmp = Path(tempfile.mkdtemp(prefix="pa_enroll_"))

    try:
        for person in args.persons:
            if person in gallery.people:  # --append replaces this person's set
                del gallery.people[person]
            photos = [
                p for p in db.photos(persons=[person])
                if p.isphoto and p.date and p.date >= cutoff and not p.ismissing
            ]
            photos.sort(key=lambda p: p.date, reverse=True)
            photos = photos[: args.candidates]
            print(f"{person}: scanning {len(photos)} recent candidate photos")
            enrolled = 0
            for p in photos:
                if enrolled >= args.per_person:
                    break
                targets = [f for f in p.face_info if f.name == person
                           and (f.quality is None or f.quality >= args.min_quality)]
                if not targets or not p.path:
                    continue
                img = _oriented_bgr(p.path, tmp)
                if img is None:
                    continue
                h, w = img.shape[:2]
                faces = rec.detect_and_embed(img)  # [((x,y,fw,fh), emb)]
                if not faces:
                    continue
                gate = (0.15 * max(w, h)) ** 2
                for f in targets:
                    if enrolled >= args.per_person:
                        break
                    tx, ty = f.center_x * w, (1.0 - f.center_y) * h  # Photos origin = bottom-left
                    best, best_d = None, float("inf")
                    for (bx, by, bw, bh), emb in faces:
                        d = (bx + bw / 2 - tx) ** 2 + (by + bh / 2 - ty) ** 2
                        if d < best_d:
                            best_d, best = d, emb
                    if best is not None and best_d <= gate:
                        gallery.enroll(person, best)
                        enrolled += 1
            print(f"  enrolled {enrolled} faces for {person}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    gallery.save(out)
    counts = {n: len(gallery.people[n]) for n in gallery.names()}
    print(f"\nsaved gallery -> {out}  (counts: {counts})")
    return 0 if gallery.names() else 1


if __name__ == "__main__":
    raise SystemExit(main())
