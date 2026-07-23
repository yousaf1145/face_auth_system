

from __future__ import annotations

import base64
import logging
import logging.handlers
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from config import get_config

_CONFIG = get_config()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def build_logger(name: str = "face_auth") -> logging.Logger:
    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(_CONFIG.logging.level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        _CONFIG.logging.log_file,
        maxBytes=_CONFIG.logging.max_bytes,
        backupCount=_CONFIG.logging.backup_count,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


LOGGER = build_logger()


# --------------------------------------------------------------------------- #
# Result / status types
# --------------------------------------------------------------------------- #
class AuthStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    SPOOF_DETECTED = "SPOOF_DETECTED"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    LOW_QUALITY_FACE = "LOW_QUALITY_FACE"
    AWAITING_MORE_FRAMES = "AWAITING_MORE_FRAMES"  # optical flow buffer still warming up


@dataclass
class AuthResult:
    """Everything the API / UI / log line needs about one auth attempt."""
    status: AuthStatus
    name: Optional[str] = None
    recognition_similarity: Optional[float] = None
    minifas_real_score: Optional[float] = None
    minifas_spoof_score: Optional[float] = None
    optical_flow_score: Optional[float] = None
    combined_liveness_score: Optional[float] = None
    reason: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def log_auth_attempt(result: AuthResult) -> None:
    """Write one structured line per authentication attempt."""
    LOGGER.info(
        "AUTH_ATTEMPT | status=%s | user=%s | rec_sim=%s | real=%s | spoof=%s | "
        "flow=%s | combined=%s | reason=%s",
        result.status.value,
        result.name or "-",
        _fmt(result.recognition_similarity),
        _fmt(result.minifas_real_score),
        _fmt(result.minifas_spoof_score),
        _fmt(result.optical_flow_score),
        _fmt(result.combined_liveness_score),
        result.reason or "-",
    )


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "-"


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def decode_base64_image(data_url_or_b64: str) -> np.ndarray:
    """
    Decode a base64 / data-URL encoded JPEG or PNG (as sent by the browser
    canvas) into a BGR numpy array suitable for OpenCV.
    """
    if "," in data_url_or_b64 and data_url_or_b64.strip().startswith("data:"):
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]

    raw = base64.b64decode(data_url_or_b64)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data — invalid or corrupt payload.")
    return image


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip a bounding box to image bounds."""
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embedding vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))
