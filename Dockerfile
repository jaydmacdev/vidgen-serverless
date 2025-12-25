# VidGen Production Dockerfile - CORRECTED
# Using actual RunPod PyTorch tags that exist

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Prevent prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (one layer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir \
    runpod==1.6.2 \
    diffusers==0.31.0 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface-hub==0.20.3 \
    safetensors==0.4.2 \
    peft==0.7.1 \
    pillow==10.2.0 \
    imageio==2.34.0 \
    imageio-ffmpeg==0.4.9 && \
    rm -rf ~/.cache/pip

# Copy handler
COPY handler.py .

# Create cache directory
RUN mkdir -p /runpod-volume/.cache

# Environment variables
ENV HF_HOME=/runpod-volume/.cache \
    TRANSFORMERS_CACHE=/runpod-volume/.cache

# Run handler
CMD ["python", "-u", "handler.py"]
