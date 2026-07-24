
from __future__ import annotations

import secrets
import threading
from typing import Dict

from flask import Flask, jsonify, request, send_from_directory, session

from pathlib import Path

from config import get_config
from pipeline import AuthenticationPipeline
from utils import LOGGER, decode_base64_image

CONFIG = get_config()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["SECRET_KEY"] = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = CONFIG.server.max_content_length_mb * 1024 * 1024

_sessions: Dict[str, AuthenticationPipeline] = {}
_sessions_lock = threading.Lock()


def _get_pipeline() -> AuthenticationPipeline:
    """Fetch (or create) the AuthenticationPipeline for the current browser session."""
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    sid = session["sid"]

    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = AuthenticationPipeline()
        return _sessions[sid]


# --------------------------------------------------------------------------- #
# Front-end
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/session/start", methods=["POST"])
def start_session():
    
    pipeline = _get_pipeline()
    pipeline.reset_session()
    return jsonify({"ok": True})


@app.route("/api/frame", methods=["POST"])
def submit_frame():
    
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing 'image' field."}), 400

    try:
        frame = decode_base64_image(image_data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pipeline = _get_pipeline()
    result = pipeline.process_frame(frame)
    return jsonify(result.to_dict())


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


def create_app() -> Flask:
    return app


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", CONFIG.server.port))

    LOGGER.info("Starting Face Auth server on 0.0.0.0:%d", port)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=CONFIG.server.debug,
        threaded=True,
    )
