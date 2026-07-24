
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent          # project root
BACKEND_DIR = Path(__file__).resolve().parent               # backend/
MODELS_DIR = BACKEND_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FaceDetectionConfig:
    """Settings for the SCRFD face detector."""
    model_pack: str = "buffalo_l"          # insightface model pack name
    detection_size: tuple[int, int] = (640, 640)
    det_score_threshold: float = 0.55      # minimum confidence to accept a detection
    ctx_id: int = -1                       # -1 = CPU, >=0 = GPU device id
    reject_multiple_faces: bool = True     # abort auth if >1 face is present
    min_face_ratio: float = 0.08           # face bbox width / frame width, filters tiny/far faces


@dataclass(frozen=True)
class AntiSpoofConfig:
    """Settings for the MiniFASNetV2 ONNX passive-anti-spoofing model."""
    model_path: str = str(MODELS_DIR / "MiniFASNetV2.onnx")
    input_size: tuple[int, int] = (80, 80)   # MiniFASNetV2 native input resolution
    crop_scale: float = 2.7                  # bbox expansion factor used by the reference implementation
    real_index: int = 1                      # index of the "real/live" class in the softmax output
    spoof_threshold: float = 0.90            # min "real" probability required to pass PAD
    high_confidence_real_score: float = 0.97 # above this, optical flow alone can no longer veto (see pipeline.py)
    intra_op_threads: int = 2


@dataclass(frozen=True)
class OpticalFlowConfig:
    """Settings for the optical-flow based replay-attack detector."""
    buffer_size: int = 6                     # number of recent frames kept for flow analysis
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    
    min_motion_energy: float = 0.08          # below this the subject looks frozen (possible static photo)
    max_uniformity_for_live: float = 0.55    # above this the motion looks too planar/uniform (possible screen replay)
    flow_pass_threshold: float = 0.3         # combined optical-flow liveness score needed to "pass"


@dataclass(frozen=True)
class LivenessFusionConfig:
    """How MiniFASNet and Optical-Flow scores are combined into one decision."""
    minifasnet_weight: float = 0.7
    optical_flow_weight: float = 0.3
    combined_pass_threshold: float = 0.80
    # Optical flow needs several frames buffered before it can produce a score.
    # Until then we fall back to MiniFASNet alone (weight = 1.0).
    min_frames_for_flow: int = 4


@dataclass(frozen=True)
class RecognitionConfig:
    """Settings for the ArcFace (buffalo_l) recognizer."""
    model_pack: str = "buffalo_l"
    embedding_dim: int = 512
    similarity_threshold: float = 0.45       # cosine similarity, tune to your dataset (0.35-0.55 typical)
    embeddings_path: str = str(MODELS_DIR / "reference_embeddings.npy")
    labels_path: str = str(MODELS_DIR / "reference_labels.pkl")
    match_strategy: str = "max"              # "max" or "mean" across a person's stored embeddings


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 25
    buffer_size: int = 1                     # keep the OS capture queue shallow to reduce latency


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    max_content_length_mb: int = 8


@dataclass(frozen=True)
class LoggingConfig:
    log_file: str = str(LOGS_DIR / "auth_events.log")
    level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


@dataclass(frozen=True)
class Config:
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    anti_spoof: AntiSpoofConfig = field(default_factory=AntiSpoofConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    liveness_fusion: LivenessFusionConfig = field(default_factory=LivenessFusionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Singleton config instance used throughout the app.
CONFIG = Config()


def get_config() -> Config:
    """Return the process-wide configuration singleton."""
    return CONFIG