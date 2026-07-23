
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import get_config
from utils import LOGGER

CONFIG = get_config()


@dataclass
class SpoofResult:
    real_score: float
    spoof_score: float
    is_real: bool


class AntiSpoofModel:
    """Thread-safe, lazily-loaded MiniFASNetV2 ONNX Runtime session."""

    _instance: Optional["AntiSpoofModel"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AntiSpoofModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._session_lock = threading.Lock()
        self._session = self._load_session()
        self._input_name = self._session.get_inputs()[0].name if self._session else None
        self._initialized = True

    @staticmethod
    def _load_session():
        import onnxruntime as ort

        model_path = Path(CONFIG.anti_spoof.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MiniFASNetV2 ONNX model not found at '{model_path}'.\n"
                "Download/convert the model and place it there before starting "
                "the server (see README.md -> Model Setup)."
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = CONFIG.anti_spoof.intra_op_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        LOGGER.info("MiniFASNetV2 anti-spoof model loaded from %s.", model_path)
        return session

    def predict(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> SpoofResult:
        """
        Run PAD inference on the face region of `frame_bgr` defined by `bbox`.
        """
        face_patch = self._crop_with_scale(frame_bgr, bbox, CONFIG.anti_spoof.crop_scale)
        tensor = self._preprocess(face_patch)

        with self._session_lock:
            outputs = self._session.run(None, {self._input_name: tensor})

        logits = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        probs = self._softmax(logits)

        real_idx = CONFIG.anti_spoof.real_index
        real_score = float(probs[real_idx])
        spoof_score = float(1.0 - real_score) if probs.size == 2 else float(max(
            p for i, p in enumerate(probs) if i != real_idx
        ))

        is_real = real_score >= CONFIG.anti_spoof.spoof_threshold
        return SpoofResult(real_score=real_score, spoof_score=spoof_score, is_real=is_real)

    @staticmethod
    def _crop_with_scale(frame: np.ndarray, bbox: tuple[int, int, int, int], scale: float) -> np.ndarray:
        """
        Expand the detected face bbox by `scale` around its center (the
        reference MiniFASNet training recipe crops a wider context than the
        tight face box) and clip to image bounds.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        cx, cy = x1 + box_w / 2.0, y1 + box_h / 2.0

        new_w, new_h = box_w * scale, box_h * scale
        nx1 = int(max(0, cx - new_w / 2.0))
        ny1 = int(max(0, cy - new_h / 2.0))
        nx2 = int(min(w, cx + new_w / 2.0))
        ny2 = int(min(h, cy + new_h / 2.0))

        if nx2 <= nx1 or ny2 <= ny1:
            return frame[y1:y2, x1:x2]  # fall back to the raw bbox
        return frame[ny1:ny2, nx1:nx2]

    @staticmethod
    def _preprocess(face_patch: np.ndarray) -> np.ndarray:
        size = CONFIG.anti_spoof.input_size
        resized = cv2.resize(face_patch, size, interpolation=cv2.INTER_LINEAR)
        
        bgr = resized.astype(np.float32)
        chw = np.transpose(bgr, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-8)