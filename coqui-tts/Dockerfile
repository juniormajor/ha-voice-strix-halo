FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV COQUI_TOS_AGREED=1
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    git espeak-ng libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y apex

# TTS ohne deps, damit PyTorch/torchvision/torchaudio nicht überschrieben werden
RUN pip install --no-cache-dir --no-deps "TTS>=0.22.0"

# Nur die TTS-Dependencies nachinstallieren, OHNE torch/torchvision/torchaudio
RUN pip install --no-cache-dir \
    coqpit trainer encodec \
    librosa soundfile pysbd \
    gruut spacy langdetect \
    "transformers>=4.33,<4.46" tokenizers \
    inflect num2words pypinyin \
    fsspec packaging hangul_romanize pydantic_settings \
    sentence_stream

RUN git clone https://github.com/lmoe/wyoming-xtts.git /app
WORKDIR /app

RUN sed -i 's/requires-python = ">=3.11"/requires-python = ">=3.10"/' pyproject.toml
RUN pip install --no-cache-dir --no-deps .

# Restliche wyoming-xtts deps
RUN pip install --no-cache-dir wyoming langdetect

EXPOSE 10200

ENTRYPOINT ["python", "-m", "wyoming_xtts", \
    "--uri", "0.0.0.0"]
