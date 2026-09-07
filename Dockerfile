# Multi-stage build for smaller final image
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Copy and install Python dependencies
COPY requirements.txt requirements-billing.txt ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Cloud (paid mode) deps: installed always so one image serves both modes; they
# are only imported when BILLING_ENABLED is set. Harmless/unused in self-host.
RUN pip install --no-cache-dir -r requirements-billing.txt

# GPU build (--build-arg GPU=1): user-space CUDA libs only — the NVIDIA
# container runtime injects the driver. cuBLAS 12 + cuDNN 9 for CTranslate2
# (faster-whisper CUDA), onnx-asr + onnxruntime-gpu for Parakeet. Adds ~2GB,
# so the default CPU image stays slim.
ARG GPU=0
RUN if [ "$GPU" = "1" ]; then \
      pip install --no-cache-dir \
        "nvidia-cublas-cu12<13" "nvidia-cudnn-cu12>=9,<10" \
        onnx-asr onnxruntime-gpu; \
    fi

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg, OpenCV deps, Node.js + npm + git (for yt-dlp JS + bgutil build).
# fontconfig + fonts-liberation back the subtitle font choices: without real
# fonts libass falls back to DejaVu for every UI option (issue #57).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    nodejs \
    npm \
    git \
    fontconfig \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Deno JS runtime — required by yt-dlp for some extractor challenges.
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

# Helper token provider, baked in as a local Node script (no separate service).
RUN git clone --depth 1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider /opt/bgutil-provider \
    && cd /opt/bgutil-provider/server \
    && npm install --no-audit --no-fund \
    && npx tsc \
    && npm cache clean --force
ENV BGUTIL_SCRIPT_PATH=/opt/bgutil-provider/server/build/generate_once.js

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# GPU runtime wiring — harmless no-ops on CPU builds / hosts without the
# NVIDIA runtime. LD_LIBRARY_PATH points at the pip-installed CUDA libs
# (paths simply don't exist in CPU images); DRIVER_CAPABILITIES asks the
# runtime for compute (CUDA) + video (NVENC) driver libs.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cu13/lib
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# Latest yt-dlp (nightly — it updates frequently) plus its helper plugin.
RUN pip install --upgrade --pre --no-cache-dir "yt-dlp[default]" bgutil-ytdlp-pot-provider

# Copy application code
COPY . .

# Register the bundled fonts (Anton for Impact; Montserrat ExtraBold, Bebas
# Neue and Bangers for the caption presets) and the UI-name -> real-font
# aliases with fontconfig so libass resolves what the subtitle modal offers.
RUN mkdir -p /usr/local/share/fonts/openshorts \
    && cp fonts/*.ttf /usr/local/share/fonts/openshorts/ \
    && cp fonts/openshorts-fontmap.conf /etc/fonts/conf.d/60-openshorts.conf \
    && fc-cache -f

# Create a non-root user (Moved up)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Create directories including Ultralytics cache config. /app/.cache/huggingface
# exists in-image (appuser-owned via the chown below) so a persistent volume
# mounted there inherits writable ownership for the ASR model downloads.
RUN mkdir -p /app/uploads /app/output /app/.cache/huggingface /tmp/Ultralytics
# Fix permissions: /app for code/uploads, /tmp/Ultralytics for AI cache
RUN chown -R appuser:appuser /app /tmp/Ultralytics

# Switch to non-root user
USER appuser

# Pre-download YOLO model on build (now running as appuser)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app. --proxy-headers + --forwarded-allow-ips trust the reverse
# proxy's X-Forwarded-Proto so generated URLs (e.g. the OAuth redirect_uri) use
# https in production instead of the internal http scheme.
# --timeout-graceful-shutdown bounds how long uvicorn waits for in-flight
# connections once app.py's drain has handed it the SIGTERM. Without it the
# default is "forever": an open range download of /api/source kept the old
# container alive for the whole 900 s stop grace period on 2026-08-25, with
# its listening socket already closed, so Traefik sent half of all requests
# to a dead port (alternating 502/200) for 15 minutes.
# Readiness for the reverse proxy: Traefik (docker provider) only routes to
# containers whose health is "healthy", so a new instance gets no traffic until
# it answers and an instance that received SIGTERM (503 from /health/ready)
# is dropped within interval*retries, while its socket is still open. Coolify's
# rolling update also waits on this before stopping the old container. The
# app is up in ~3 s; start-period covers slow disks. curl is installed above
# for this: when the Coolify health check is enabled it replaces this
# HEALTHCHECK with its own curl/wget command, and an image without either
# reports unhealthy forever and every deploy rolls back (2026-08-25).
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=2 \
  CMD curl -sf http://127.0.0.1:8000/health/ready > /dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-graceful-shutdown", "15"]
