"""Local face detection + recognition (OpenCV YuNet + SFace).

YuNet detects faces; SFace produces 128-d embeddings. A `FaceGallery` stores
per-person embeddings and matches by cosine similarity. Used to recognize
enrolled people (e.g. family) in a frame — fully on-device, no cloud. Models
auto-download from the OpenCV Zoo on first use.

The gallery stores only embeddings (not images); embeddings are not reversible
to photos, which is the right hygiene for family/biometric data.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_REPO_MODELS = Path(__file__).resolve().parents[3] / "models"  # repo-root models/ (gitignored)
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_URLS = {
    YUNET: f"{_ZOO}/face_detection_yunet/{YUNET}",
    SFACE: f"{_ZOO}/face_recognition_sface/{SFACE}",
}
# OpenCV's recommended same-identity cosine threshold for SFace.
SFACE_COSINE_THRESHOLD = 0.363


def ensure_models(models_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (yunet_path, sface_path), downloading them if missing."""
    d = models_dir or _REPO_MODELS
    d.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in _URLS.items():
        p = d / name
        if not p.exists() or p.stat().st_size < 1000:
            log.info("downloading face model %s", name)
            try:
                urllib.request.urlretrieve(url, p)
                if p.stat().st_size < 1000:  # tiny body = HTML error page, not a model
                    raise RuntimeError(f"downloaded face model looks invalid: {name}")
            except Exception:
                p.unlink(missing_ok=True)  # never cache a partial/bad download
                raise
        paths[name] = p
    return paths[YUNET], paths[SFACE]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


class FaceGallery:
    """name -> (N, 128) enrolled embeddings; match by best cosine similarity."""

    def __init__(self) -> None:
        self.people: dict[str, np.ndarray] = {}

    def enroll(self, name: str, embedding) -> None:
        e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if name in self.people:
            self.people[name] = np.vstack([self.people[name], e])
        else:
            self.people[name] = e

    def match(
        self, embedding, threshold: float = SFACE_COSINE_THRESHOLD
    ) -> tuple[str | None, float]:
        """Return (name, score) for the best match, or (None, best_score) if below threshold."""
        emb = np.asarray(embedding, dtype=np.float32)
        best_name, best = None, -1.0
        for name, embs in self.people.items():
            for row in embs:
                s = _cosine(emb, row)
                if s > best:
                    best, best_name = s, name
        return (best_name, best) if best >= threshold else (None, best)

    def names(self) -> list[str]:
        return list(self.people)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.people)
        arrays: dict[str, np.ndarray] = {f"emb_{i}": self.people[n] for i, n in enumerate(names)}
        arrays["__names__"] = np.array(names, dtype=object)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> FaceGallery:
        g = cls()
        with np.load(path, allow_pickle=True) as data:
            for i, n in enumerate(list(data["__names__"])):
                g.people[str(n)] = data[f"emb_{i}"]
        return g


class FaceRecognizer:
    """YuNet detect + SFace embed; identify enrolled faces in a frame."""

    def __init__(self, models_dir: Path | None = None,
                 det_score_threshold: float = 0.7, nms_threshold: float = 0.3):
        import cv2

        yunet, sface = ensure_models(models_dir)
        self._det = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), det_score_threshold, nms_threshold, 5000)
        self._rec = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect_and_embed(self, image) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        if faces is None:
            return out
        for row in faces:
            aligned = self._rec.alignCrop(image, row)
            feat = self._rec.feature(aligned).flatten().astype(np.float32)
            x, y, fw, fh = (int(v) for v in row[:4])
            out.append(((x, y, fw, fh), feat))
        return out

    def identify(self, image, gallery: FaceGallery,
                 threshold: float = SFACE_COSINE_THRESHOLD,
                 ) -> list[tuple[tuple[int, int, int, int], str | None, float]]:
        results = []
        for bbox, emb in self.detect_and_embed(image):
            name, score = gallery.match(emb, threshold)
            results.append((bbox, name, score))
        return results
