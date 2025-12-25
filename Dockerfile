# VidGen Serverless - No tzdata issues
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install only essential dependencies (skip tzdata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install dependencies one by one to see which fails
RUN pip install --no-cache-dir runpod==1.6.2 && \
    pip install --no-cache-dir diffusers==0.31.0 && \
    pip install --no-cache-dir transformers==4.36.2 && \
    pip install --no-cache-dir accelerate==0.25.0 && \
    pip install --no-cache-dir huggingface-hub==0.20.3 && \
    pip install --no-cache-dir safetensors==0.4.2 && \
    pip install --no-cache-dir peft==0.7.1 && \
    pip install --no-cache-dir pillow==10.2.0 && \
    pip install --no-cache-dir imageio==2.34.0 && \
    pip install --no-cache-dir imageio-ffmpeg==0.4.9

COPY handler.py .

CMD ["python", "-u", "handler.py"]
