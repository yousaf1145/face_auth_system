
from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import get_config
from utils import LOGGER, cosine_similarity

CONFIG = get_config()


@dataclass
class MatchResult:
    name: Optional[str]
    similarity: float
    matched: bool


class FaceRecognizer:
    """Thread-safe, lazily-loaded ArcFace embedding extractor + matcher."""

    _instance: Optional["FaceRecognizer"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FaceRecognizer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model_lock = threading.Lock()
        self._model = self._load_model()
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[list[str]] = None
        self.reload_gallery()
        self._initialized = True

    @staticmethod
    def _load_model():
        try:
            from insightface.model_zoo import get_model
            from insightface.utils import ensure_available
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "insightface is not installed. Run `pip install insightface onnxruntime`."
            ) from exc

        
        model_dir = ensure_available("models", CONFIG.recognition.model_pack, root="~/.insightface")
        rec_model_path = str(Path(model_dir) / "w600k_r50.onnx")
        model = get_model(rec_model_path, providers=["CPUExecutionProvider"])
        model.prepare(ctx_id=-1)
        LOGGER.info("ArcFace recognition model loaded (pack=%s).", CONFIG.recognition.model_pack)
        return model

    def reload_gallery(self) -> None:
        """(Re)load the enrolled embeddings + labels from disk."""
        emb_path = Path(CONFIG.recognition.embeddings_path)
        lbl_path = Path(CONFIG.recognition.labels_path)

        if not emb_path.exists() or not lbl_path.exists():
            LOGGER.warning(
                "No enrolled gallery found (%s / %s). Run enroll.py first.", emb_path, lbl_path
            )
            self._embeddings = np.zeros((0, CONFIG.recognition.embedding_dim), dtype=np.float32)
            self._labels = []
            return

        self._embeddings = np.load(emb_path).astype(np.float32)
        with open(lbl_path, "rb") as f:
            self._labels = pickle.load(f)

        LOGGER.info(
            "Loaded gallery: %d embeddings across %d unique identities.",
            len(self._labels), len(set(self._labels)),
        )

    def get_embedding(self, aligned_face_bgr) -> np.ndarray:
        """Extract a 512-d embedding from a (roughly) cropped face image."""
        with self._model_lock:
            embedding = self._model.get_feat(aligned_face_bgr)
        return np.asarray(embedding).reshape(-1).astype(np.float32)

    def match(self, embedding: np.ndarray) -> MatchResult:
        """
        Compare an embedding against every enrolled identity and return the
        best match, subject to the configured similarity threshold.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return MatchResult(name=None, similarity=0.0, matched=False)

        sims = np.array([cosine_similarity(embedding, ref) for ref in self._embeddings])

        if CONFIG.recognition.match_strategy == "mean":
            per_identity: dict[str, list[float]] = {}
            for label, sim in zip(self._labels, sims):
                per_identity.setdefault(label, []).append(sim)
            averaged = {k: float(np.mean(v)) for k, v in per_identity.items()}
            best_name = max(averaged, key=averaged.get)
            best_sim = averaged[best_name]
        else:  # "max" — most permissive, most common in production kiosks
            best_idx = int(np.argmax(sims))
            best_name = self._labels[best_idx]
            best_sim = float(sims[best_idx])

        matched = best_sim >= CONFIG.recognition.similarity_threshold
        return MatchResult(name=best_name if matched else None, similarity=best_sim, matched=matched)
