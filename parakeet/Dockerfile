FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies first (preserves ROCm PyTorch)
RUN pip install --no-cache-dir \
    wyoming \
    numpy \
    soundfile \
    librosa \
    Cython \
    hydra-core \
    omegaconf \
    sentencepiece \
    braceexpand \
    editdistance \
    jiwer \
    lhotse \
    webdataset \
    huggingface_hub \
    lightning \
    pytorch-lightning \
    torchmetrics

# Install NeMo toolkit without pulling in its own PyTorch
# --no-deps avoids overwriting the ROCm PyTorch already in the image
RUN pip install --no-cache-dir --no-deps \
    nemo_toolkit[asr]

# Create data directory for model cache
RUN mkdir -p /data/models

# Copy application files
COPY parakeet_wrapper.py /app/parakeet_wrapper.py
COPY parakeet_handler.py /app/parakeet_handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Wyoming protocol port
EXPOSE 10303

ENTRYPOINT ["/app/entrypoint.sh"]
