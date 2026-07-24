
`# SENTRY – Face Recognition & Anti-Spoofing Access System

SENTRY is a real-time AI-powered face authentication system that combines face recognition with liveness detection to prevent spoofing attacks such as printed photos and mobile screen replays.

## Features

- Real-time Face Detection (SCRFD)
- ArcFace Face Recognition
- MiniFASNetV2 Anti-Spoofing
- Optical Flow Liveness Verification
- Flask Web Application
- CPU-based Inference
- Easy Docker Deployment

---

## Tech Stack

- Python 3.11
- Flask
- InsightFace
- ONNX Runtime
- OpenCV
- NumPy
- Docker

---

## Project Structure

```
face_auth_system/
│
├── backend/
│   ├── main.py
│   ├── enroll.py
│   ├── models/
│   └── ...
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── dataset/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Run Without Docker

## 1. Clone Repository

```bash
git clone https://github.com/yousaf1145/face_auth_system.git

cd face_auth_system
```

## 2. Create Virtual Environment

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Enroll Dataset

Place images like

```
dataset/
    person1/
        img1.jpg
        img2.jpg

    person2/
        img1.jpg
```

Generate face embeddings

```bash
cd backend

python enroll.py --images ../dataset --output models
```

---

## 5. Start Application

```bash
cd backend
python main.py
```

Open

```
http://localhost:8001
```

---

# Docker Deployment

## Build Image

```bash
docker build -t your-dockerhub-username/face-auth-system:latest .
```

Example

```bash
docker build -t yousafkhan1165/face-auth-system:latest .
```

---

## Run Container

```bash
docker run -it -p 8001:8001 --name face-auth-container your-dockerhub-username/face-auth-system:latest
```

Example

```bash
docker run -it -p 8001:8001 --name face-auth-container yousafkhan1165/face-auth-system:latest
```

Open

```
http://localhost:8001
```

---

## Stop Container

```bash
docker stop face-auth-container
```

---

## Start Existing Container

```bash
docker start -ai face-auth-container
```

---

## Remove Container

```bash
docker rm -f face-auth-container
```

---

## Remove Image

```bash
docker rmi yousafkhan1165/face-auth-system:latest
```

---

# Models

The project automatically downloads the InsightFace `buffalo_l` model during the Docker build.

Required model:

```
backend/models/MiniFASNetV2.onnx
```

Generated after enrollment:

```
backend/models/reference_embeddings.npy

backend/models/reference_labels.pkl
```

---

# Authentication Pipeline

```
Camera
    │
    ▼
SCRFD Face Detection
    │
    ▼
MiniFASNetV2 Anti-Spoofing
    │
    ▼
Optical Flow Verification
    │
    ▼
ArcFace Recognition
    │
    ▼
Authorized / Rejected
```

---

# License

This project is intended for educational and portfolio purposes.