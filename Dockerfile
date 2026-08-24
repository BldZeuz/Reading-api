FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    VOSK_MODEL_PATH=/opt/vosk-model \
    PRELOAD_MODEL=1

ARG VOSK_MODEL_URL=https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
ARG VOSK_MODEL_DIR=vosk-model-small-en-us-0.15

WORKDIR /app


# ============================================================
# SYSTEM PACKAGES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       ffmpeg \
       unzip \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON PACKAGES
# ============================================================

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt


# ============================================================
# DOWNLOAD VOSK SMALL ENGLISH MODEL
# ============================================================

RUN curl \
    --fail \
    --location \
    --retry 3 \
    "$VOSK_MODEL_URL" \
    -o /tmp/vosk-model.zip \
    && unzip -q /tmp/vosk-model.zip -d /opt \
    && mv "/opt/${VOSK_MODEL_DIR}" "$VOSK_MODEL_PATH" \
    && rm /tmp/vosk-model.zip


# ============================================================
# APPLICATION
# ============================================================

COPY main.py .


# ============================================================
# START API
# ============================================================

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]