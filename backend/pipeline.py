
from __future__ import annotations

import numpy as np

from anti_spoof import AntiSpoofModel
from config import get_config
from face_detector import FaceDetector, MultipleFacesError
from optical_flow import OpticalFlowAnalyzer
from recognizer import FaceRecognizer
from utils import AuthResult, AuthStatus, log_auth_attempt

CONFIG = get_config()


class AuthenticationPipeline:
   

    def __init__(self) -> None:
        # Heavy models are process-wide singletons (see their `__new__`);
        # only the optical-flow buffer is per-session state.
        self._detector = FaceDetector()
        self._anti_spoof = AntiSpoofModel()
        self._recognizer = FaceRecognizer()
        self._flow = OpticalFlowAnalyzer()

    def reset_session(self) -> None:
        """Call when a new authentication attempt begins (e.g. new browser session)."""
        self._flow.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> AuthResult:
        
        # 1. Face detection ---------------------------------------------------
        try:
            face = self._detector.detect(frame_bgr)
        except MultipleFacesError:
            result = AuthResult(
                status=AuthStatus.MULTIPLE_FACES_DETECTED,
                reason="More than one face detected in frame; only a single subject may authenticate.",
            )
            log_auth_attempt(result)
            return result

        if face is None:
            result = AuthResult(
                status=AuthStatus.NO_FACE_DETECTED,
                reason="No qualifying face found in frame.",
            )
            log_auth_attempt(result)
            return result

        # 2. MiniFASNetV2 anti-spoof (primary PAD) ----------------------------
        spoof = self._anti_spoof.predict(frame_bgr, face.bbox)

        # 3. Optical flow replay detection (secondary, temporal signal) ------
        flow = self._flow.update(frame_bgr, face.bbox)

        # 4. Fuse both liveness signals into one decision ---------------------
        fusion = CONFIG.liveness_fusion
        if flow.ready:
            combined = (
                fusion.minifasnet_weight * spoof.real_score
                + fusion.optical_flow_weight * flow.score
            )
        else:
            # Optical flow buffer still warming up — rely on MiniFASNet alone
            # but do not yet grant access; ask caller to keep streaming frames.
            combined = spoof.real_score

        is_spoof_from_pad = not spoof.is_real
        
        is_spoof_from_flow = (
            flow.ready
            and flow.score < CONFIG.optical_flow.flow_pass_threshold
            and spoof.real_score < CONFIG.anti_spoof.high_confidence_real_score
        )
        liveness_passed = (
            spoof.is_real
            and combined >= fusion.combined_pass_threshold
            and not is_spoof_from_flow
        )

        if is_spoof_from_pad:
            # Stop immediately — face recognition must NEVER run on a spoof.
            result = AuthResult(
                status=AuthStatus.SPOOF_DETECTED,
                minifas_real_score=spoof.real_score,
                minifas_spoof_score=spoof.spoof_score,
                optical_flow_score=flow.score if flow.ready else None,
                combined_liveness_score=combined,
                reason="MiniFASNetV2 flagged the presented face as a spoof (print/screen).",
            )
            log_auth_attempt(result)
            return result

        if not flow.ready:
            result = AuthResult(
                status=AuthStatus.AWAITING_MORE_FRAMES,
                minifas_real_score=spoof.real_score,
                minifas_spoof_score=spoof.spoof_score,
                optical_flow_score=None,
                combined_liveness_score=combined,
                reason="Optical-flow motion buffer warming up; keep streaming frames.",
            )
            log_auth_attempt(result)
            return result

        if not liveness_passed:
            result = AuthResult(
                status=AuthStatus.SPOOF_DETECTED,
                minifas_real_score=spoof.real_score,
                minifas_spoof_score=spoof.spoof_score,
                optical_flow_score=flow.score,
                combined_liveness_score=combined,
                reason="Optical-flow motion pattern looked planar/uniform (possible screen replay).",
            )
            log_auth_attempt(result)
            return result

        # 5. Liveness passed -> ArcFace recognition (never reached otherwise) -
        embedding = self._recognizer.get_embedding(face.embedding_input)
        match = self._recognizer.match(embedding)

        if not match.matched:
            result = AuthResult(
                status=AuthStatus.UNKNOWN_PERSON,
                recognition_similarity=match.similarity,
                minifas_real_score=spoof.real_score,
                minifas_spoof_score=spoof.spoof_score,
                optical_flow_score=flow.score,
                combined_liveness_score=combined,
                reason="Live face confirmed but did not match any enrolled identity.",
            )
            log_auth_attempt(result)
            return result

        result = AuthResult(
            status=AuthStatus.AUTHORIZED,
            name=match.name,
            recognition_similarity=match.similarity,
            minifas_real_score=spoof.real_score,
            minifas_spoof_score=spoof.spoof_score,
            optical_flow_score=flow.score,
            combined_liveness_score=combined,
            reason="Live face verified and matched an enrolled identity.",
        )
        log_auth_attempt(result)
        return result