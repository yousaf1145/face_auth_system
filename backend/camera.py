
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import get_config
from utils import LOGGER

CONFIG = get_config()


class ThreadedCamera:
    """Background-thread webcam reader with the latest-frame-wins semantics."""

    def __init__(self, device_index: Optional[int] = None) -> None:
        self._device_index = device_index if device_index is not None else CONFIG.camera.device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedCamera":
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self._device_index}.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera.frame_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG.camera.buffer_size)
        self._cap.set(cv2.CAP_PROP_FPS, CONFIG.camera.target_fps)

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        LOGGER.info("Camera %s started.", self._device_index)
        return self

    def _update_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        LOGGER.info("Camera %s stopped.", self._device_index)

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
