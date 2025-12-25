# VidGen Production Dockerfile - Optimized for RunPod
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

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
    imageio-ffmpeg==0.4.9 \
    xformers==0.0.23.post1

# Copy handler
COPY handler.py .

# Create cache directory
RUN mkdir -p /runpod-volume/.cache

# Set environment variables
ENV HF_HOME=/runpod-volume/.cache
ENV TRANSFORMERS_CACHE=/runpod-volume/.cache
ENV HF_DATASETS_CACHE=/runpod-volume/.cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Run handler
CMD ["python", "-u", "handler.py"]
