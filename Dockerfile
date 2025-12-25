# VidGen Serverless - Simple & Reliable
# Use official PyTorch image with CUDA pre-installed

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
