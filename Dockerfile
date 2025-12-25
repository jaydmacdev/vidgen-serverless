# VidGen Production Dockerfile
# ✅ Correct base image tag
# ✅ Compatible dependency versions
# ✅ Optimized for GitHub Actions

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/.cache \
    TRANSFORMERS_CACHE=/runpod-volume/.cache

WORKDIR /app

# Install system dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements
COPY requirements.txt .

# Install Python packages - ALL COMPATIBLE VERSIONS
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        runpod==1.6.2 \
        diffusers==0.31.0 \
        transformers==4.36.2 \
        accelerate==0.31.0 \
        huggingface-hub==0.26.2 \
        safetensors==0.4.5 \
        peft==0.13.2 \
        pillow==10.2.0 \
        imageio==2.34.0 \
        imageio-ffmpeg==0.4.9 && \
    rm -rf ~/.cache/pip /tmp/*

# Copy application
COPY handler.py .

# Create cache directory
RUN mkdir -p /runpod-volume/.cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Metadata
LABEL maintainer="vidgen" \
      version="1.0" \
      description="VidGen Wan2.2 I2V Serverless"

# Run handler
CMD ["python", "-u", "handler.py"]
