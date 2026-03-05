FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git .

# torch aus requirements heraushalten — kommt aus dem Base-Image (2.9.1+ROCm 7.2)
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm7.2 \
    || pip install -e ".[server]" --extra-index-url https://download.pytorch.org/whl/rocm7.2

EXPOSE 8880

ENV TTS_BACKEND=official
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8880

CMD ["python", "-m", "api.main"]
