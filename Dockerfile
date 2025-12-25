# VidGen Production - Wan2.2 I2V with 4-Step Distilled LoRA
# Optimized for RunPod Serverless deployment

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/.cache \
    TRANSFORMERS_CACHE=/runpod-volume/.cache \
    TORCH_HOME=/runpod-volume/.cache

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir \
        runpod==1.6.2 \
        torch==2.4.0 \
        diffusers==0.31.0 \
        transformers==4.36.2 \
        accelerate==0.31.0 \
        huggingface-hub==0.26.2 \
        safetensors==0.4.5 \
        peft==0.13.2 \
        pillow==10.2.0 \
        imageio==2.34.0 \
        imageio-ffmpeg==0.4.9 \
        sentencepiece==0.2.0 && \
    rm -rf ~/.cache/pip /tmp/*

# Copy handler
COPY handler.py .

# Create cache directories
RUN mkdir -p /runpod-volume/.cache

# Pre-download models (optional - speeds up cold starts)
# Uncomment to pre-cache models in image (increases image size)
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('Wan-AI/Wan2.2-I2V-A14B', local_dir='/runpod-volume/.cache/Wan2.2-I2V-A14B'); \
#     snapshot_download('lightx2v/Wan2.2-Distill-Loras', local_dir='/runpod-volume/.cache/Wan2.2-Distill-Loras')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Metadata
LABEL maintainer="vidgen" \
      version="2.0" \
      description="VidGen Wan2.2 I2V 4-Step" \
      model="Wan2.2-I2V-A14B" \
      lora="4-step-distilled"

# Run handler
CMD ["python", "-u", "handler.py"]
