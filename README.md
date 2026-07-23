# SENTRY — Face Recognition & Anti-Spoofing Access System

A CPU-only, real-time face authentication system:

```
Camera → SCRFD Face Detection → MiniFASNetV2 Anti-Spoof → Optical Flow
       → ArcFace Recognition → Authorized
```

Two front-ends are included:
- **`backend/desktop_app.py`** — a native OpenCV window, for kiosks/local testing.
- **`backend/main.py`** — a Flask API + the `frontend/` HTML/JS 3-step web UI
  (Identify → Verify Liveness → Access), with a "Login" button on success and
  alert popups on **Unknown Person** / **Spoof Detected**.

Your dataset (`dataset/yousaf/*.jpg`) has already been added to the project
so `enroll.py` can enroll that identity out of the box.

---

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested against Python 3.10/3.11 on CPU-only Intel i5 hardware.

## 2. Model setup

### 2.1 ArcFace + SCRFD (automatic)
The first time the app runs, `insightface` automatically downloads the
`buffalo_l` model pack (SCRFD detector + ArcFace `w600k_r50` recognizer) to
`~/.insightface/models/buffalo_l/`. No manual step needed — just make sure
the machine has internet access the first time.

### 2.2 MiniFASNetV2 (manual — required)
MiniFASNetV2 weights are **not bundled** in this project (they're a
separate third-party model, not distributable here). Get an ONNX export of
MiniFASNetV2 from the Silent-Face-Anti-Spoofing project family and place it
at:

```
backend/models/MiniFASNetV2.onnx
```

Notes:
- If you only have the original `.pth`/MiniVision checkpoint, convert it to
  ONNX first (`torch.onnx.export`) — a static 80×80×3 input, single output
  of shape `[1, N]` (N=2 or 3 classes) works with `anti_spoof.py` as-is.
- If your exported model uses 3 output classes instead of 2, update
  `AntiSpoofConfig.real_index` in `config.py` to match the "real" class index
  (`anti_spoof.py` already handles both cases).
- Until this file exists, `main.py` / `desktop_app.py` will raise a clear
  `FileNotFoundError` on startup telling you exactly what's missing.

## 3. Enroll your dataset

```bash
cd backend
python enroll.py --images ../dataset/ --output models/
```

This walks `dataset/<person_name>/*.jpg`, detects a face in each image with
SCRFD, extracts a 512-d ArcFace embedding, and writes:

- `backend/models/reference_embeddings.npy`
- `backend/models/reference_labels.pkl`

Images with no face, more than one face, or that fail to load are skipped
with a warning — enrollment never crashes on a single bad file. Your
`dataset/yousaf/` folder will enroll as the identity **"yousaf"**; that is
the *only* identity that will ever be recognized as Authorized until you add
more folders under `dataset/`.

## 4. Run it

**Desktop (OpenCV window):**
```bash
cd backend
python desktop_app.py
```
Press `q` to quit, `r` to reset the liveness buffer between subjects.

**Web UI (Flask + HTML/JS):**
```bash
cd backend
python main.py
```
Open `http://localhost:8001` in a browser. Flow:
1. **Identify** — click "Recognize" to grant camera access.
2. **Verify Liveness** — click "Start Liveness Check"; the terminal streams
   frames to the backend, showing REAL / SPOOF / motion-flow / combined
   scores live.
3. **Access** — Authorized → **Login** button. Unknown person or spoof
   (printed photo, phone/tablet photo, replay video) → a red/amber alert
   popup and access denied.

## 5. Configuration

Everything tunable lives in `backend/config.py` — recognition/PAD
thresholds, camera resolution, optical-flow parameters, liveness fusion
weights, logging. No source-code changes needed to retune for a new camera
or environment.

## 6. Logging

Every authentication attempt (desktop or web) is appended to
`logs/auth_events.log`: timestamp, user, recognition similarity, MiniFASNet
real/spoof scores, optical-flow score, final decision, and rejection reason.

## 7. How replay/photo spoofing is caught

1. **MiniFASNetV2** (primary): a single-frame texture/frequency classifier
   trained to tell real skin from print/screen material. If it says "spoof,"
   the pipeline stops immediately — recognition never runs.
2. **Optical flow** (secondary, temporal): a real 3-D face produces
   non-uniform motion (parallax, micro-expressions); a photo or screen held
   up to the camera moves as one flat, rigid rectangle. The system watches a
   short rolling buffer of frames and computes both motion energy (catches
   frozen photos) and motion uniformity (catches rigid replay). This score
   is fused with MiniFASNet's before a final liveness decision is made.

**Honest limitation:** software-only, RGB-camera liveness detection — with
no depth/IR sensor — cannot be made bulletproof against a very high-quality
replay (e.g., a high-res OLED screen with a matching frame rate and someone
physically moving the device to fake parallax). This system meaningfully
raises the bar against printed photos, casual phone-screen replays, and
static/rigid video replays, which is what was requested — but for a truly
high-security deployment, pair this with a depth or infrared camera and
periodic re-verification.

## 8. Project structure

```
face_auth_system/
├── backend/
│   ├── main.py            # Flask API + web UI server
│   ├── desktop_app.py      # standalone OpenCV real-time app
│   ├── pipeline.py         # orchestrates the full auth pipeline (single source of truth)
│   ├── face_detector.py    # SCRFD wrapper
│   ├── anti_spoof.py       # MiniFASNetV2 ONNX wrapper
│   ├── optical_flow.py     # replay-attack motion analysis
│   ├── recognizer.py       # ArcFace embeddings + matching
│   ├── camera.py           # threaded webcam capture
│   ├── enroll.py           # dataset -> embeddings CLI
│   ├── config.py           # all tunables
│   ├── utils.py            # logging, result types, image helpers
│   └── models/             # MiniFASNetV2.onnx + generated gallery files go here
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── dataset/
│   └── yousaf/              # your provided photos
├── logs/
├── requirements.txt
└── README.md
```
