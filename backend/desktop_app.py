
from __future__ import annotations

import time

import cv2

from camera import ThreadedCamera
from config import get_config
from pipeline import AuthenticationPipeline
from utils import AuthStatus, LOGGER

CONFIG = get_config()

STATUS_COLORS = {
    AuthStatus.AUTHORIZED: (0, 200, 0),
    AuthStatus.UNKNOWN_PERSON: (0, 165, 255),
    AuthStatus.SPOOF_DETECTED: (0, 0, 255),
    AuthStatus.NO_FACE_DETECTED: (128, 128, 128),
    AuthStatus.MULTIPLE_FACES_DETECTED: (0, 165, 255),
    AuthStatus.LOW_QUALITY_FACE: (128, 128, 128),
    AuthStatus.AWAITING_MORE_FRAMES: (255, 200, 0),
}


def draw_overlay(frame, result, fps: float) -> None:
    color = STATUS_COLORS.get(result.status, (255, 255, 255))
    label = result.status.value.replace("_", " ")
    if result.name:
        label = f"{label}: {result.name}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    telemetry = (
        f"REAL={_fmt(result.minifas_real_score)} "
        f"SPOOF={_fmt(result.minifas_spoof_score)} "
        f"FLOW={_fmt(result.optical_flow_score)} "
        f"SIM={_fmt(result.recognition_similarity)} "
        f"FPS={fps:.1f}"
    )
    cv2.putText(frame, telemetry, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _fmt(value) -> str:
    return f"{value:.2f}" if value is not None else "-"


def main() -> None:
    pipeline = AuthenticationPipeline()
    window_name = "Face Auth — press 'q' to quit, 'r' to reset liveness buffer"

    with ThreadedCamera() as camera:
        # Give the capture thread a moment to deliver its first frame.
        time.sleep(0.5)

        fps = 0.0
        last_time = time.time()

        while True:
            frame = camera.read()
            if frame is None:
                continue

            result = pipeline.process_frame(frame)

            now = time.time()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
            last_time = now

            draw_overlay(frame, result, fps)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                pipeline.reset_session()
                LOGGER.info("Liveness buffer reset by operator.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
