
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

from config import get_config

CONFIG = get_config()


@dataclass
class FlowResult:
    score: float                 # combined liveness-likeness in [0, 1]
    motion_energy: float
    uniformity: float
    ready: bool                  # False while the buffer is still warming up


class OpticalFlowAnalyzer:
   
    def __init__(self) -> None:
        self._buffer: Deque[np.ndarray] = deque(maxlen=CONFIG.optical_flow.buffer_size)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()

    def update(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> FlowResult:
        
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return FlowResult(score=0.0, motion_energy=0.0, uniformity=1.0, ready=False)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)

        with self._lock:
            self._buffer.append(gray)
            frames = list(self._buffer)

        cfg = CONFIG.optical_flow
        if len(frames) < 2:
            return FlowResult(score=0.0, motion_energy=0.0, uniformity=1.0, ready=False)

        energies = []
        uniformities = []
        for prev, curr in zip(frames[:-1], frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=cfg.pyr_scale,
                levels=cfg.levels,
                winsize=cfg.winsize,
                iterations=cfg.iterations,
                poly_n=cfg.poly_n,
                poly_sigma=cfg.poly_sigma,
                flags=cfg.flags,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_mag = float(np.mean(magnitude))
            std_mag = float(np.std(magnitude))
            # Coefficient of variation: low value = very uniform (planar) motion.
            uniformity = std_mag / (mean_mag + 1e-6)
            energies.append(mean_mag)
            uniformities.append(uniformity)

        motion_energy = float(np.mean(energies))
        # Normalize uniformity into [0, 1] "non-uniformity score" where higher
        # is more consistent with a real, non-planar face.
        raw_uniformity = float(np.mean(uniformities))
        non_planarity = min(raw_uniformity / 1.5, 1.0)  # empirical scaling, tune per camera

        has_motion = min(motion_energy / max(cfg.min_motion_energy, 1e-6), 1.0)

        ready = len(frames) >= CONFIG.liveness_fusion.min_frames_for_flow
        score = float(np.clip(0.5 * has_motion + 0.5 * non_planarity, 0.0, 1.0))

        return FlowResult(
            score=score,
            motion_energy=motion_energy,
            uniformity=raw_uniformity,
            ready=ready,
        )
