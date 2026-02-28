FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install all NeMo dependencies explicitly (preserves ROCm PyTorch)
# Core requirements (requirements.txt)
RUN pip install --no-cache-dir \
    fsspec \
    huggingface_hub>=0.24 \
    numba \
    "numexpr<2.14.0" \
    numpy>=1.22 \
    "onnx>=1.7.0" \
    "protobuf~=5.29.5" \
    python-dateutil \
    ruamel.yaml \
    scikit-learn \
    tensorboard \
    text-unidecode \
    "tqdm>=4.41.0" \
    "setuptools>=70.0.0" \
    wget \
    wrapt

# Lightning requirements (requirements_lightning.txt)
RUN pip install --no-cache-dir \
    "cloudpickle" \
    fiddle \
    "hydra-core>1.3,<=1.3.2" \
    "lightning>2.2.1,<=2.4.0" \
    "omegaconf<=2.3" \
    peft \
    "torchmetrics>=0.11.0" \
    "transformers~=4.57.0" \
    webdataset>=0.2.86 \
    nv_one_logger_core>=2.3.1 \
    nv_one_logger_training_telemetry>=2.3.1 \
    nv_one_logger_pytorch_lightning_integration>=2.3.1 \
    wandb

# ASR requirements (requirements_asr.txt)
RUN pip install --no-cache-dir \
    braceexpand \
    diskcache \
    editdistance \
    einops \
    "jiwer>=3.1.0,<4.0.0" \
    kaldi-python-io \
    "kaldialign<=0.9.1" \
    "lhotse>=1.32.2" \
    "librosa>=0.10.1" \
    marshmallow \
    optuna \
    packaging \
    pyannote.core \
    pyannote.metrics \
    pydub \
    pyloudnorm \
    resampy \
    sacrebleu \
    "scipy>=0.14" \
    soundfile \
    "sox<=1.5.0" \
    whisper_normalizer

# Common requirements (requirements_common.txt)
RUN pip install --no-cache-dir \
    datasets \
    inflect \
    pandas \
    mediapy==1.1.6 \
    "sacremoses>=0.0.43" \
    "sentencepiece<1.0.0"

# Wyoming protocol
RUN pip install --no-cache-dir wyoming

# Install NeMo toolkit without pulling in its own PyTorch
# --no-deps avoids overwriting the ROCm PyTorch already in the image
RUN pip install --no-cache-dir --no-deps nemo_toolkit[asr]

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
