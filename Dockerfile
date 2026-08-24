FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    HF_HOME=/opt/hf-cache

ARG WAV2VEC2_MODEL=facebook/wav2vec2-base-960h

ENV WAV2VEC2_MODEL=${WAV2VEC2_MODEL}

# ------------------------------------------------------------
# Install system dependencies
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------------------------
# Install Python packages
# ------------------------------------------------------------

COPY requirements.txt .

# CPU-only PyTorch
RUN pip install --upgrade pip \
    && pip install torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# ------------------------------------------------------------
# Download Wav2Vec2 during Docker build
#
# This prevents the first API request from having to download
# hundreds of MB of model data.
# ------------------------------------------------------------

RUN python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('${WAV2VEC2_MODEL}')"

# ------------------------------------------------------------
# Application
# ------------------------------------------------------------

COPY main.py .

# Render provides the PORT environment variable automatically.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]