# Dockerfile for Whisper STT Local Server
# Ubuntu 24.04 + torch cu128 (CUDA runtime bundled in wheel — no nvidia base needed)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps: python3.12 is the default in 24.04
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (torch >=2.9.0,<2.10.0 pinned in requirements.txt)
COPY requirements.txt .
RUN python3.12 -m venv venv && \
    ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Model cache lives in a volume — create the directory structure
RUN mkdir -p assets/models/whisper assets/cache

EXPOSE 5000

CMD ["./venv/bin/uvicorn", "main_stt:app", "--host", "0.0.0.0", "--port", "5000"]
