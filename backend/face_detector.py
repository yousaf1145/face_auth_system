
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import get_config
from utils import LOGGER, clamp_bbox

CONFIG = get_config()


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2 in pixel coords
    kps: np.ndarray                     # 5-point landmarks, shape (5, 2)
    det_score: float
    embedding_input: np.ndarray         # aligned/cropped face ready for downstream models (BGR)


class MultipleFacesError(Exception):
    """Raised when more than one qualifying face is present and the policy forbids it."""


class FaceDetector:
   

    _instance: Optional["FaceDetector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FaceDetector":
        # Simple thread-safe singleton so the (relatively heavy) ONNX model
        # is loaded exactly once per process.
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
        self._app = self._load_model()
        self._initialized = True

    @staticmethod
    def _load_model():
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "insightface is not installed. Run `pip install insightface onnxruntime`."
            ) from exc

        app = FaceAnalysis(
            name=CONFIG.face_detection.model_pack,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(
            ctx_id=CONFIG.face_detection.ctx_id,
            det_size=CONFIG.face_detection.detection_size,
            det_thresh=CONFIG.face_detection.det_score_threshold,
        )
        LOGGER.info("SCRFD face detector loaded (pack=%s).", CONFIG.face_detection.model_pack)
        return app

    def detect(self, frame_bgr: np.ndarray, enforce_single_face: Optional[bool] = None) -> Optional[DetectedFace]:
        
        if enforce_single_face is None:
            enforce_single_face = CONFIG.face_detection.reject_multiple_faces

        h, w = frame_bgr.shape[:2]
        with self._model_lock:
            faces = self._app.get(frame_bgr)

        qualifying = [f for f in faces if self._qualifies(f, w, h)]

        if not qualifying:
            return None

        if enforce_single_face and len(qualifying) > 1:
            raise MultipleFacesError(f"{len(qualifying)} qualifying faces detected.")

        # Largest face by bbox area.
        best = max(qualifying, key=lambda f: self._bbox_area(f.bbox))
        x1, y1, x2, y2 = clamp_bbox(*[int(v) for v in best.bbox], width=w, height=h)
        crop = frame_bgr[y1:y2, x1:x2]

        return DetectedFace(
            bbox=(x1, y1, x2, y2),
            kps=best.kps.astype(np.float32),
            det_score=float(best.det_score),
            embedding_input=crop,
        )

    def _qualifies(self, face, frame_w: int, frame_h: int) -> bool:
        x1, y1, x2, y2 = face.bbox
        width_ratio = (x2 - x1) / float(frame_w)
        return (
            float(face.det_score) >= CONFIG.face_detection.det_score_threshold
            and width_ratio >= CONFIG.face_detection.min_face_ratio
        )

    @staticmethod
    def _bbox_area(bbox) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
