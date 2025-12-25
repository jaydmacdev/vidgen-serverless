# VidGen Serverless - Simplified Dockerfile
# Using RunPod's official PyTorch base image

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="VidGen Serverless - Image to Video Generation API"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]
