# ----------------------------
# Base Image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Python Environment
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------
# Install Linux Packages
# ----------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install Python Packages
# ----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy Project
# ----------------------------
COPY . .

# ----------------------------
# Pre-download InsightFace Model
# ----------------------------
RUN python -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1)"

# ----------------------------
# Move to Backend
# ----------------------------
WORKDIR /app/backend

# ----------------------------
# Flask Port
# ----------------------------
EXPOSE 8001

# ----------------------------
# Start Application
# ----------------------------
CMD ["python", "main.py"]